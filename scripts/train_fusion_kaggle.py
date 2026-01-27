# KAGGLE'DA ÇALIŞTIR - Fusion Model Training
# CLIP + FFT Frequency Features

!pip install torch transformers datasets pillow tqdm -q

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

# Weighted Loss
REAL_WEIGHT = 3.0
FAKE_WEIGHT = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# FREQUENCY ANALYZER
# ============================================================
class FrequencyAnalyzer(nn.Module):
    """FFT-based frequency feature extractor"""
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

# ============================================================
# CLIP HEAD (trained)
# ============================================================
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
    
    def get_features(self, x):
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)
        x = self.classifier[6](x)
        return x

# ============================================================
# FUSION CLASSIFIER
# ============================================================
class FusionClassifier(nn.Module):
    def __init__(self, clip_dim=256, freq_dim=256):
        super().__init__()
        combined_dim = clip_dim + freq_dim
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

# ============================================================
# DATASET
# ============================================================
class ImageDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["Image"].convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), item["Label_A"]

# ============================================================
# LOAD MODELS
# ============================================================
print("="*60)
print("FUSION MODEL TRAINING")
print("="*60)

# CLIP backbone (frozen)
print("\nLoading CLIP backbone...")
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip.eval()
for p in clip.parameters():
    p.requires_grad = False

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# CLIP head (load trained weights, frozen)
print("Loading trained CLIP head...")
clip_head = CLIPHead().to(device)
CLIP_WEIGHTS = "modern_ai_detector.pt"  # Kaggle path'ini güncelle
clip_head.load_state_dict(torch.load(CLIP_WEIGHTS, map_location=device))
clip_head.eval()
for p in clip_head.parameters():
    p.requires_grad = False
print(f"✓ Loaded CLIP head from: {CLIP_WEIGHTS}")

# Frequency analyzer (trainable)
print("Initializing frequency analyzer...")
freq_analyzer = FrequencyAnalyzer(output_dim=256).to(device)

# Fusion classifier (trainable)
print("Initializing fusion classifier...")
fusion_classifier = FusionClassifier().to(device)

# ============================================================
# DATASET
# ============================================================
print("\nLoading dataset...")
ds = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")
print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}")

train_loader = DataLoader(
    ImageDataset(ds["train"], processor),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
val_loader = DataLoader(
    ImageDataset(ds["validation"], processor),
    batch_size=BATCH_SIZE,
    num_workers=2
)

# ============================================================
# TRAINING SETUP
# ============================================================
# Sadece trainable parametreler
trainable_params = list(freq_analyzer.parameters()) + list(fusion_classifier.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

weights = torch.tensor([REAL_WEIGHT, FAKE_WEIGHT]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

trainable_count = sum(p.numel() for p in trainable_params)
print(f"\nTrainable parameters: {trainable_count:,}")
print(f"Weighted Loss: Real={REAL_WEIGHT}, Fake={FAKE_WEIGHT}")

best_acc = 0

# ============================================================
# TRAINING
# ============================================================
print("\n" + "="*60)
print("STARTING FUSION TRAINING")
print("="*60)

for epoch in range(1, EPOCHS + 1):
    freq_analyzer.train()
    fusion_classifier.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # CLIP features (frozen)
        with torch.no_grad():
            clip_features = clip.get_image_features(pixel_values=images)
            semantic_features = clip_head.get_features(clip_features)
        
        # Frequency features (trainable)
        freq_features = freq_analyzer(images)
        
        # Fusion
        logits = fusion_classifier(semantic_features, freq_features)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.1f}%'})
    
    scheduler.step()
    train_acc = 100 * correct / total
    
    # Validation
    freq_analyzer.eval()
    fusion_classifier.eval()
    
    val_correct = 0
    val_total = 0
    real_correct, real_total = 0, 0
    fake_correct, fake_total = 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            clip_features = clip.get_image_features(pixel_values=images)
            semantic_features = clip_head.get_features(clip_features)
            freq_features = freq_analyzer(images)
            logits = fusion_classifier(semantic_features, freq_features)
            
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            real_mask = labels == 0
            fake_mask = labels == 1
            real_total += real_mask.sum().item()
            fake_total += fake_mask.sum().item()
            real_correct += ((predicted == labels) & real_mask).sum().item()
            fake_correct += ((predicted == labels) & fake_mask).sum().item()
    
    val_acc = 100 * val_correct / val_total
    real_acc = 100 * real_correct / real_total if real_total > 0 else 0
    fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch} COMPLETE")
    print(f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
    print(f"  Real Acc: {real_acc:.1f}% | Fake Acc: {fake_acc:.1f}%")
    
    # Save
    save_dict = {
        'freq_analyzer': freq_analyzer.state_dict(),
        'fusion_classifier': fusion_classifier.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }
    torch.save(save_dict, f"fusion_model_epoch{epoch}.pt")
    print(f"✓ Saved: fusion_model_epoch{epoch}.pt")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(save_dict, "fusion_model_best.pt")
        print(f"★ NEW BEST! {val_acc:.1f}%")

print("\n" + "="*60)
print(f"TRAINING COMPLETE! Best: {best_acc:.1f}%")
print("="*60)

from IPython.display import FileLink
display(FileLink("fusion_model_best.pt"))
