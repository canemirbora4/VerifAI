# ============================================================
# VCT² COCO_AI Training Script - Kaggle Optimized
# Continue from Epoch 8 with Balanced 50/50 Sampling
# Dual GPU Support + Mixed Precision for Maximum Speed
# ============================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 64          # Per GPU batch size (will be 128 total with 2 GPUs)
EPOCHS = 10              # Number of epochs to train
LR = 3e-4                # Learning rate (slightly lower for fine-tuning)
START_EPOCH = 9          # Continue from epoch 8 (next is 9)
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4          # DataLoader workers per GPU
SEED = 42

# Balanced sampling: Her epoch'ta her real image için 1 random fake seç
# Bu sayede tam 50/50 denge sağlanır

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Device: {device}, GPUs: {num_gpus}")

# ============================================================
# MODEL DEFINITION (Same architecture as before)
# ============================================================
class ClassificationHead(nn.Module):
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

# ============================================================
# BALANCED DATASET - 50/50 Real/Fake
# ============================================================
class VCT2BalancedDataset(Dataset):
    """
    Her örnek için: 1 real (coco_image) + 1 random fake (6 seçenekten biri)
    Bu sayede tam 50/50 denge sağlanır.
    Total: 10,017 real + 10,017 fake = 20,034 images per epoch
    """
    def __init__(self, hf_dataset, processor, split='train'):
        self.data = hf_dataset[split]
        self.processor = processor
        self.fake_columns = [
            'sd35_image', 'sd3_image', 'sd21_image', 
            'sdxl_image', 'dalle_image', 'midjourney_image'
        ]
        
        # Her örnek için (real, fake) pair oluştur
        self.samples = []
        for idx in range(len(self.data)):
            # Real image
            self.samples.append((idx, 'coco_image', 0))  # label=0 for real
            # Random fake (her epoch'ta farklı olabilir)
            fake_col = random.choice(self.fake_columns)
            self.samples.append((idx, fake_col, 1))  # label=1 for fake
        
        # Shuffle samples
        random.shuffle(self.samples)
        print(f"Dataset: {len(self.samples)} samples (50% real, 50% fake)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data_idx, col_name, label = self.samples[idx]
        
        try:
            image = self.data[data_idx][col_name]
            if image is None:
                # Fallback: use coco_image if None
                image = self.data[data_idx]['coco_image']
                label = 0
            
            image = image.convert('RGB')
            inputs = self.processor(images=image, return_tensors='pt')
            pixel_values = inputs['pixel_values'].squeeze(0)
            return pixel_values, label
        except Exception as e:
            # Fallback for any error
            image = self.data[0]['coco_image'].convert('RGB')
            inputs = self.processor(images=image, return_tensors='pt')
            return inputs['pixel_values'].squeeze(0), 0

# ============================================================
# VALIDATION DATASET - All images for proper evaluation
# ============================================================
class VCT2ValidationDataset(Dataset):
    """Validation için tüm resimleri kullan (daha doğru metrikler için)"""
    def __init__(self, hf_dataset, processor, split='train', val_ratio=0.1):
        full_data = hf_dataset[split]
        
        # Son %10'u validation olarak ayır
        total = len(full_data)
        val_start = int(total * (1 - val_ratio))
        
        self.processor = processor
        self.samples = []
        
        fake_columns = [
            'sd35_image', 'sd3_image', 'sd21_image', 
            'sdxl_image', 'dalle_image', 'midjourney_image'
        ]
        
        # Validation set için tüm resimleri ekle
        for idx in range(val_start, total):
            # Real
            self.samples.append((idx, 'coco_image', 0))
            # All fakes
            for fake_col in fake_columns:
                self.samples.append((idx, fake_col, 1))
        
        self.data = full_data
        print(f"Validation: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data_idx, col_name, label = self.samples[idx]
        try:
            image = self.data[data_idx][col_name]
            if image is None:
                image = self.data[0]['coco_image']
                label = 0
            image = image.convert('RGB')
            inputs = self.processor(images=image, return_tensors='pt')
            return inputs['pixel_values'].squeeze(0), label
        except:
            image = self.data[0]['coco_image'].convert('RGB')
            inputs = self.processor(images=image, return_tensors='pt')
            return inputs['pixel_values'].squeeze(0), 0

# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    print("="*60)
    print("VCT² COCO_AI Training - Continue from Epoch 8")
    print("="*60)
    
    # Load CLIP
    print("\n[1/5] Loading CLIP model...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Freeze CLIP backbone
    clip.eval()
    for param in clip.parameters():
        param.requires_grad = False
    
    # Multi-GPU support
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        clip = nn.DataParallel(clip)
    clip = clip.to(device)
    
    # Load classification head from epoch 8
    print("\n[2/5] Loading classification head from Epoch 8...")
    head = ClassificationHead()
    
    # IMPORTANT: Update this path to your Kaggle input path
    WEIGHTS_PATH = "/kaggle/input/your-weights/modern_ai_detector_epoch8.pt"
    # Or if you uploaded to the notebook:
    # WEIGHTS_PATH = "modern_ai_detector_epoch8.pt"
    
    if os.path.exists(WEIGHTS_PATH):
        head.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        print(f"Loaded weights from {WEIGHTS_PATH}")
    else:
        print(f"WARNING: {WEIGHTS_PATH} not found! Starting from scratch.")
        print("Please upload modern_ai_detector_epoch8.pt to Kaggle")
    
    if num_gpus > 1:
        head = nn.DataParallel(head)
    head = head.to(device)
    
    # Load dataset
    print("\n[3/5] Loading VCT² COCO_AI dataset...")
    dataset = load_dataset("NasrinImp/COCO_AI")
    
    # Create datasets
    train_dataset = VCT2BalancedDataset(dataset, processor, split='train')
    val_dataset = VCT2ValidationDataset(dataset, processor, split='train', val_ratio=0.1)
    
    # DataLoaders - Optimized for speed
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE * max(1, num_gpus),  # Scale with GPUs
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * max(1, num_gpus),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Optimizer & Scheduler
    print("\n[4/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        head.parameters() if num_gpus <= 1 else head.module.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Loss - No weighting needed (dataset is balanced 50/50)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed Precision Scaler for faster training
    scaler = GradScaler('cuda')
    
    # Training loop
    print("\n[5/5] Starting training...")
    print(f"Batch size: {BATCH_SIZE * max(1, num_gpus)} (total)")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("="*60)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        current_epoch = START_EPOCH + epoch
        
        # ==================== TRAINING ====================
        head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {current_epoch} [Train]")
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast('cuda'):
                with torch.no_grad():
                    if num_gpus > 1:
                        features = clip.module.get_image_features(pixel_values=pixel_values)
                    else:
                        features = clip.get_image_features(pixel_values=pixel_values)
                
                logits = head(features)
                loss = criterion(logits, labels)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.1f}%'
            })
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # ==================== VALIDATION ====================
        head.eval()
        val_correct = 0
        val_total = 0
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0
        
        with torch.no_grad():
            for pixel_values, labels in tqdm(val_loader, desc=f"Epoch {current_epoch} [Val]"):
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                
                with autocast('cuda'):
                    if num_gpus > 1:
                        features = clip.module.get_image_features(pixel_values=pixel_values)
                    else:
                        features = clip.get_image_features(pixel_values=pixel_values)
                    
                    logits = head(features)
                
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Per-class accuracy
                real_mask = labels == 0
                fake_mask = labels == 1
                real_correct += (preds[real_mask] == labels[real_mask]).sum().item()
                real_total += real_mask.sum().item()
                fake_correct += (preds[fake_mask] == labels[fake_mask]).sum().item()
                fake_total += fake_mask.sum().item()
        
        val_acc = 100 * val_correct / val_total
        real_acc = 100 * real_correct / real_total if real_total > 0 else 0
        fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
        
        scheduler.step()
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Epoch {current_epoch} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val Acc: {val_acc:.1f}% | Real Acc: {real_acc:.1f}% | Fake Acc: {fake_acc:.1f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*60}\n")
        
        # Save model
        save_head = head.module if num_gpus > 1 else head
        torch.save(save_head.state_dict(), f"vct2_detector_epoch{current_epoch}.pt")
        print(f"Saved: vct2_detector_epoch{current_epoch}.pt")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(save_head.state_dict(), "vct2_detector_best.pt")
            print(f"New best model! Val Acc: {val_acc:.1f}%")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()
