"""
Test Fusion Detector on Defactify Dataset
CLIP-only vs Fusion karşılaştırması
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# MODELS (same as fusion_detector.py)
# ============================================================
class FrequencyAnalyzer(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
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
        batch_size = images.shape[0]
        features = []
        for i in range(batch_size):
            img = images[i].mean(dim=0).cpu().numpy()
            img_resized = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((64, 64)))
            img_resized = img_resized.astype(np.float32) / 255.0
            fft = np.fft.fft2(img_resized)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.log1p(np.abs(fft_shifted))
            magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
            features.append(magnitude.flatten())
        return torch.tensor(np.array(features), dtype=torch.float32, device=images.device)
    
    def forward(self, images):
        fft_features = self.extract_fft_features(images)
        return self.fc(fft_features)


class CLIPHead(nn.Module):
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
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)
        x = self.classifier[6](x)
        return x


class FusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
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


# ============================================================
# TEST
# ============================================================
def main():
    project_root = Path(__file__).parent.parent
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("TESTING: CLIP-only vs Fusion (untrained)")
    print("="*60)
    print(f"Device: {device}")
    
    # Load models
    print("\nLoading CLIP backbone...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    print("Loading CLIP head...")
    clip_head = CLIPHead().to(device)
    weights_path = project_root / "models" / "modern_ai_detector.pt"
    clip_head.load_state_dict(torch.load(weights_path, map_location=device))
    clip_head.eval()
    print(f"✓ Loaded: {weights_path}")
    
    print("Initializing frequency analyzer (untrained)...")
    freq_analyzer = FrequencyAnalyzer().to(device)
    freq_analyzer.eval()
    
    print("Initializing fusion classifier (untrained)...")
    fusion_classifier = FusionClassifier().to(device)
    fusion_classifier.eval()
    
    # Check for trained fusion weights
    fusion_weights_path = project_root / "models" / "fusion_model_best.pt"
    if fusion_weights_path.exists():
        print(f"\n✓ Found trained fusion weights!")
        checkpoint = torch.load(fusion_weights_path, map_location=device)
        freq_analyzer.load_state_dict(checkpoint['freq_analyzer'])
        fusion_classifier.load_state_dict(checkpoint['fusion_classifier'])
        print(f"  Loaded from epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.1f}%")
    else:
        print(f"\n⚠ No trained fusion weights found at {fusion_weights_path}")
        print("  Testing with untrained fusion (random baseline)")
    
    # Load dataset
    print("\nLoading Defactify TEST set...")
    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")
    test_data = dataset["test"]
    print(f"Test set size: {len(test_data)}")
    
    # Sample
    n_samples = 500
    indices = random.sample(range(len(test_data)), min(n_samples, len(test_data)))
    
    # Track results
    clip_results = {"correct": 0, "total": 0, "real_correct": 0, "real_total": 0, 
                   "fake_correct": 0, "fake_total": 0, "scores": []}
    fusion_results = {"correct": 0, "total": 0, "real_correct": 0, "real_total": 0,
                     "fake_correct": 0, "fake_total": 0, "scores": []}
    
    print(f"\nTesting {n_samples} images...")
    
    for idx in tqdm(indices, desc="Testing"):
        item = test_data[idx]
        image = item["Image"].convert("RGB")
        label = item["Label_A"]  # 0=real, 1=fake
        
        # Preprocess
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        with torch.no_grad():
            # CLIP features
            clip_features = clip.get_image_features(pixel_values=pixel_values)
            
            # CLIP-only prediction
            clip_logits = clip_head(clip_features)
            clip_probs = torch.softmax(clip_logits, dim=1)
            clip_ai_prob = clip_probs[0, 1].item()
            clip_pred = 1 if clip_ai_prob > 0.5 else 0
            
            # Fusion prediction
            semantic_features = clip_head.get_features(clip_features)
            freq_features = freq_analyzer(pixel_values)
            fusion_logits = fusion_classifier(semantic_features, freq_features)
            fusion_probs = torch.softmax(fusion_logits, dim=1)
            fusion_ai_prob = fusion_probs[0, 1].item()
            fusion_pred = 1 if fusion_ai_prob > 0.5 else 0
        
        # Update CLIP results
        clip_results["total"] += 1
        clip_results["correct"] += int(clip_pred == label)
        clip_results["scores"].append((clip_ai_prob, label))
        if label == 0:
            clip_results["real_total"] += 1
            clip_results["real_correct"] += int(clip_pred == label)
        else:
            clip_results["fake_total"] += 1
            clip_results["fake_correct"] += int(clip_pred == label)
        
        # Update Fusion results
        fusion_results["total"] += 1
        fusion_results["correct"] += int(fusion_pred == label)
        fusion_results["scores"].append((fusion_ai_prob, label))
        if label == 0:
            fusion_results["real_total"] += 1
            fusion_results["real_correct"] += int(fusion_pred == label)
        else:
            fusion_results["fake_total"] += 1
            fusion_results["fake_correct"] += int(fusion_pred == label)
    
    # Calculate metrics
    def calc_metrics(results):
        acc = 100 * results["correct"] / results["total"]
        real_acc = 100 * results["real_correct"] / results["real_total"] if results["real_total"] > 0 else 0
        fake_acc = 100 * results["fake_correct"] / results["fake_total"] if results["fake_total"] > 0 else 0
        
        real_scores = [s for s, l in results["scores"] if l == 0]
        fake_scores = [s for s, l in results["scores"] if l == 1]
        real_mean = np.mean(real_scores) if real_scores else 0
        fake_mean = np.mean(fake_scores) if fake_scores else 0
        
        return {
            "accuracy": acc,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
            "real_mean": real_mean,
            "fake_mean": fake_mean,
            "separation": fake_mean - real_mean
        }
    
    clip_metrics = calc_metrics(clip_results)
    fusion_metrics = calc_metrics(fusion_results)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"Samples: {clip_results['real_total']} real, {clip_results['fake_total']} fake")
    
    print(f"\n{'Metric':<20} {'CLIP-only':<15} {'Fusion':<15} {'Diff':<10}")
    print("-"*60)
    print(f"{'Accuracy':<20} {clip_metrics['accuracy']:.1f}%{'':<10} {fusion_metrics['accuracy']:.1f}%{'':<10} {fusion_metrics['accuracy']-clip_metrics['accuracy']:+.1f}%")
    print(f"{'Real Accuracy':<20} {clip_metrics['real_acc']:.1f}%{'':<10} {fusion_metrics['real_acc']:.1f}%{'':<10} {fusion_metrics['real_acc']-clip_metrics['real_acc']:+.1f}%")
    print(f"{'Fake Accuracy':<20} {clip_metrics['fake_acc']:.1f}%{'':<10} {fusion_metrics['fake_acc']:.1f}%{'':<10} {fusion_metrics['fake_acc']-clip_metrics['fake_acc']:+.1f}%")
    print(f"{'Separation':<20} {clip_metrics['separation']:.3f}{'':<11} {fusion_metrics['separation']:.3f}{'':<11} {fusion_metrics['separation']-clip_metrics['separation']:+.3f}")
    
    print("\n" + "="*60)
    if fusion_metrics['accuracy'] > clip_metrics['accuracy']:
        print("✓ FUSION IS BETTER!")
    elif fusion_metrics['accuracy'] < clip_metrics['accuracy']:
        print("⚠ CLIP-only is better (fusion needs training)")
    else:
        print("= Same performance")
    print("="*60)


if __name__ == "__main__":
    random.seed(42)
    main()
