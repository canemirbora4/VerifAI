"""
FUSION AI DETECTOR
==================
CLIP (semantic features) + FFT (frequency artifacts) kombinasyonu

AI-generated görüntüler frequency domain'de belirli izler bırakır:
- Diffusion modelleri: Belirli frekanslarda pattern
- GAN'lar: Checkerboard artifacts
- Upscaling: High-frequency anomalies
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import warnings
warnings.filterwarnings("ignore")


class FrequencyAnalyzer(nn.Module):
    """
    Görüntüden frequency-domain features çıkarır.
    FFT kullanarak AI üretim izlerini yakalar.
    """
    def __init__(self, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        
        # FFT features -> dense features
        # 64x64 FFT magnitude = 4096 features
        self.fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def extract_fft_features(self, images):
        """
        Batch of images -> FFT magnitude features
        images: [B, C, H, W] tensor
        """
        batch_size = images.shape[0]
        features = []
        
        for i in range(batch_size):
            # RGB -> Grayscale
            img = images[i].mean(dim=0)  # [H, W]
            
            # Resize to 64x64 for consistent FFT size
            img_np = img.cpu().numpy()
            img_resized = np.array(Image.fromarray((img_np * 255).astype(np.uint8)).resize((64, 64)))
            img_resized = img_resized.astype(np.float32) / 255.0
            
            # 2D FFT
            fft = np.fft.fft2(img_resized)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.log1p(np.abs(fft_shifted))  # Log magnitude
            
            # Normalize
            magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
            
            features.append(magnitude.flatten())
        
        return torch.tensor(np.array(features), dtype=torch.float32, device=images.device)
    
    def forward(self, images):
        fft_features = self.extract_fft_features(images)  # [B, 4096]
        return self.fc(fft_features)  # [B, output_dim]


class CLIPHead(nn.Module):
    """
    Mevcut eğitilmiş CLIP classification head
    """
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.classifier(x)
    
    def get_features(self, x):
        """Son katmandan önceki features (256-dim)"""
        x = self.classifier[0](x)  # Linear 768->512
        x = self.classifier[1](x)  # LayerNorm
        x = self.classifier[2](x)  # GELU
        x = self.classifier[3](x)  # Dropout
        x = self.classifier[4](x)  # Linear 512->256
        x = self.classifier[5](x)  # LayerNorm
        x = self.classifier[6](x)  # GELU
        return x  # [B, 256]


class FusionClassifier(nn.Module):
    """
    CLIP features + Frequency features -> Final prediction
    """
    def __init__(self, clip_dim=256, freq_dim=256):
        super().__init__()
        
        # Fusion MLP
        combined_dim = clip_dim + freq_dim  # 512
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, clip_features, freq_features):
        combined = torch.cat([clip_features, freq_features], dim=1)
        return self.fusion(combined)


class FusionDetector(nn.Module):
    """
    Complete Fusion Detector
    
    Pipeline:
    1. Image -> CLIP backbone -> CLIP features (768)
    2. CLIP features -> Trained head -> Semantic features (256)
    3. Image -> FFT -> Frequency features (256)
    4. Concat [Semantic, Frequency] -> Fusion MLP -> Prediction
    """
    def __init__(self, clip_weights_path=None, device='cpu'):
        super().__init__()
        self.device = device
        
        # CLIP backbone (frozen)
        print("Loading CLIP backbone...")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False
        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # CLIP classification head (load trained weights)
        print("Loading CLIP head...")
        self.clip_head = CLIPHead()
        if clip_weights_path:
            self.clip_head.load_state_dict(torch.load(clip_weights_path, map_location=device))
            print(f"✓ Loaded CLIP head from: {clip_weights_path}")
        
        # Freeze CLIP head (use as feature extractor)
        for p in self.clip_head.parameters():
            p.requires_grad = False
        
        # Frequency analyzer (trainable)
        print("Initializing frequency analyzer...")
        self.freq_analyzer = FrequencyAnalyzer(output_dim=256)
        
        # Fusion classifier (trainable)
        print("Initializing fusion classifier...")
        self.fusion_classifier = FusionClassifier(clip_dim=256, freq_dim=256)
        
        self.to(device)
        print("✓ Fusion Detector ready!")
    
    def forward(self, pixel_values):
        """
        pixel_values: [B, 3, 224, 224] preprocessed images
        """
        # CLIP features
        with torch.no_grad():
            clip_features = self.clip.get_image_features(pixel_values=pixel_values)  # [B, 768]
            semantic_features = self.clip_head.get_features(clip_features)  # [B, 256]
        
        # Frequency features
        freq_features = self.freq_analyzer(pixel_values)  # [B, 256]
        
        # Fusion
        logits = self.fusion_classifier(semantic_features, freq_features)  # [B, 2]
        
        return logits
    
    def predict(self, image):
        """
        Single image prediction
        image: PIL Image or path
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(pixel_values)
            probs = torch.softmax(logits, dim=1)
            ai_prob = probs[0, 1].item()
        
        return {
            "ai_probability": ai_prob,
            "prediction": "AI-generated" if ai_prob > 0.5 else "Real",
            "confidence": max(ai_prob, 1 - ai_prob)
        }
    
    def get_trainable_params(self):
        """Sadece trainable parametreleri döndür"""
        params = []
        params.extend(self.freq_analyzer.parameters())
        params.extend(self.fusion_classifier.parameters())
        return params


def create_fusion_detector(clip_weights_path, device='cpu'):
    """
    Factory function to create fusion detector
    """
    return FusionDetector(clip_weights_path=clip_weights_path, device=device)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    from pathlib import Path
    
    # Paths
    project_root = Path(__file__).parent.parent
    weights_path = project_root / "models" / "modern_ai_detector.pt"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create detector
    detector = create_fusion_detector(
        clip_weights_path=str(weights_path),
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in detector.parameters())
    trainable_params = sum(p.numel() for p in detector.get_trainable_params())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with dummy input
    print("\nTesting with dummy input...")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = detector(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    print("\n✓ Fusion Detector working!")
