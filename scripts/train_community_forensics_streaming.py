# ============================================================
# CommunityForensics Training Script - STREAMING VERSION
# No disk download required - streams data directly
# Works within Kaggle's resource limits
# ============================================================

import os
import io
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 32
EPOCHS = 5                   # Fewer epochs for streaming (can't easily re-iterate)
LR = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 2              # Lower for streaming
SEED = 42

# Samples per epoch (streaming doesn't know total size)
TRAIN_SAMPLES_PER_EPOCH = 50000   # 25K real + 25K fake per epoch
VAL_SAMPLES = 5000                # 2.5K real + 2.5K fake

# Anti-shortcut: JPEG compression range
JPEG_QUALITY_MIN = 70
JPEG_QUALITY_MAX = 100

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Device: {device}, GPUs: {num_gpus}")

# ============================================================
# MODEL DEFINITION
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
# ANTI-SHORTCUT PREPROCESSING
# ============================================================
def anti_shortcut_preprocess(image: Image.Image, is_train: bool = True) -> Image.Image:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if is_train:
        quality = random.randint(JPEG_QUALITY_MIN, JPEG_QUALITY_MAX)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        image = Image.open(buffer).convert('RGB')
        
        if random.random() < 0.3:
            interp_methods = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
            interp = random.choice(interp_methods)
            w, h = image.size
            scale = random.uniform(0.9, 1.1)
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, interp)
            image = image.resize((w, h), interp)
    else:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        image = Image.open(buffer).convert('RGB')
    
    return image

def decode_image(sample) -> Image.Image:
    if 'image' in sample and sample['image'] is not None:
        img = sample['image']
        if isinstance(img, Image.Image):
            return img.convert('RGB')
        elif isinstance(img, bytes):
            return Image.open(io.BytesIO(img)).convert('RGB')
    if 'image_data' in sample and sample['image_data'] is not None:
        return Image.open(io.BytesIO(sample['image_data'])).convert('RGB')
    raise ValueError("Could not decode image")

# ============================================================
# STREAMING DATASET (No disk storage needed)
# ============================================================
class CommunityForensicsStreaming(IterableDataset):
    """
    Streaming dataset that doesn't download to disk.
    Balances real/fake on-the-fly.
    """
    def __init__(self, processor, max_samples, is_train=True):
        self.processor = processor
        self.max_samples = max_samples
        self.is_train = is_train
        
        print(f"Initializing streaming dataset (max {max_samples} samples)...")
        
        # Load in streaming mode - NO DISK DOWNLOAD
        self.stream = load_dataset(
            "OwensLab/CommunityForensics-Small",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    
    def __iter__(self):
        real_count = 0
        fake_count = 0
        target_per_class = self.max_samples // 2
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Seed each worker differently
            worker_seed = SEED + worker_info.id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
        
        for sample in self.stream:
            # Check if we've collected enough
            if real_count >= target_per_class and fake_count >= target_per_class:
                break
            
            try:
                label = int(sample.get('label', 1))
                
                # Balance: skip if we have enough of this class
                if label == 0 and real_count >= target_per_class:
                    continue
                if label == 1 and fake_count >= target_per_class:
                    continue
                
                # Decode and preprocess
                image = decode_image(sample)
                image = anti_shortcut_preprocess(image, is_train=self.is_train)
                
                # Process for CLIP
                inputs = self.processor(images=image, return_tensors='pt')
                pixel_values = inputs['pixel_values'].squeeze(0)
                
                if label == 0:
                    real_count += 1
                else:
                    fake_count += 1
                
                yield pixel_values, label
                
            except Exception as e:
                continue
        
        print(f"  Streamed: {real_count} real + {fake_count} fake")

# ============================================================
# CLIP FORWARD HELPER
# ============================================================
def get_clip_features(model, pixel_values):
    out = model(pixel_values=pixel_values, return_dict=False)
    
    if len(out) > 3 and torch.is_tensor(out[3]) and out[3].ndim == 2 and out[3].shape[-1] == 768:
        return out[3]
    
    matches = [t for t in out if torch.is_tensor(t) and t.ndim == 2 and t.shape[-1] == 768]
    if matches:
        return matches[-1]
    
    raise RuntimeError(f"Could not locate image_embeds")

# ============================================================
# EXTERNAL VALIDATION (Defactify)
# ============================================================
def evaluate_on_defactify(model, head, processor, device, n_samples=50):
    print("\n  Evaluating on Defactify (OOD)...")
    
    try:
        ds = load_dataset(
            "Rajarshi-Roy-research/Defactify_Image_Dataset",
            split="test",
            streaming=True  # Stream this too
        )
    except Exception as e:
        print(f"  Could not load Defactify: {e}")
        return 0, 0, 0
    
    model.eval()
    head.eval()
    
    real_correct, real_total = 0, 0
    ai_correct, ai_total = 0, 0
    
    with torch.no_grad():
        for sample in ds:
            if real_total >= n_samples and ai_total >= n_samples:
                break
            
            try:
                img = sample['Image'].convert('RGB')
                true_label = sample['Label_A']
                
                if true_label == 0 and real_total >= n_samples:
                    continue
                if true_label == 1 and ai_total >= n_samples:
                    continue
                
                inputs = processor(images=img, return_tensors='pt')
                pixel_values = inputs['pixel_values'].to(device)
                
                features = get_clip_features(model, pixel_values)
                logits = head(features)
                pred = torch.argmax(logits, dim=1).item()
                
                if true_label == 0:
                    real_total += 1
                    real_correct += (pred == 0)
                else:
                    ai_total += 1
                    ai_correct += (pred == 1)
                    
            except Exception:
                continue
    
    real_acc = real_correct / max(1, real_total)
    ai_acc = ai_correct / max(1, ai_total)
    overall = (real_correct + ai_correct) / max(1, real_total + ai_total)
    
    return real_acc, ai_acc, overall

# ============================================================
# TRAINING LOOP
# ============================================================
def train():
    print("="*60)
    print("CommunityForensics Training - STREAMING VERSION")
    print("No disk download required!")
    print("="*60)
    
    # Load CLIP
    print("\nLoading CLIP ViT-L/14...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    for param in model.parameters():
        param.requires_grad = False
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"CLIP wrapped in DataParallel ({num_gpus} GPUs)")
    
    model = model.to(device)
    model.eval()
    print("CLIP loaded and frozen!")
    
    # Fresh classification head
    head = ClassificationHead()
    print(f"Classification head parameters: {sum(p.numel() for p in head.parameters()):,}")
    
    if num_gpus > 1:
        head = nn.DataParallel(head)
    head = head.to(device)
    
    # Optimizer
    head_params = head.module.parameters() if hasattr(head, 'module') else head.parameters()
    optimizer = torch.optim.AdamW(head_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    best_defactify_real = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Create fresh streaming dataset for each epoch
        train_dataset = CommunityForensicsStreaming(
            processor, 
            max_samples=TRAIN_SAMPLES_PER_EPOCH,
            is_train=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE * max(1, num_gpus),
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        
        # Train
        head.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_real_correct, train_real_total = 0, 0
        train_ai_correct, train_ai_total = 0, 0
        
        pbar = tqdm(train_loader, desc="Training", total=TRAIN_SAMPLES_PER_EPOCH // (BATCH_SIZE * max(1, num_gpus)))
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                with torch.no_grad():
                    features = get_clip_features(model, pixel_values)
                logits = head(features)
                loss = criterion(logits.float(), labels.long())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            for p, l in zip(preds, labels):
                if l.item() == 0:
                    train_real_total += 1
                    train_real_correct += (p.item() == 0)
                else:
                    train_ai_total += 1
                    train_ai_correct += (p.item() == 1)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/max(1,train_total):.1f}%'
            })
        
        train_acc = train_correct / max(1, train_total)
        train_real_acc = train_real_correct / max(1, train_real_total)
        train_ai_acc = train_ai_correct / max(1, train_ai_total)
        
        print(f"\nTrain Acc: {100*train_acc:.2f}%")
        print(f"  Real: {100*train_real_acc:.1f}% ({train_real_correct}/{train_real_total})")
        print(f"  AI:   {100*train_ai_acc:.1f}% ({train_ai_correct}/{train_ai_total})")
        
        # Validation on Defactify (external OOD)
        def_real, def_ai, def_overall = evaluate_on_defactify(
            model, head, processor, device, n_samples=50
        )
        print(f"\nDefactify (OOD):")
        print(f"  Real: {100*def_real:.1f}%, AI: {100*def_ai:.1f}%, Overall: {100*def_overall:.1f}%")
        
        # Save best model
        if def_real > best_defactify_real:
            best_defactify_real = def_real
            save_head = head.module if hasattr(head, 'module') else head
            torch.save(save_head.state_dict(), 'cf_detector_best.pt')
            print(f"  ★ New best model! Defactify Real: {100*def_real:.1f}%")
        
        # Save checkpoint
        save_head = head.module if hasattr(head, 'module') else head
        torch.save(save_head.state_dict(), f'cf_detector_epoch{epoch}.pt')
        print(f"  Checkpoint: cf_detector_epoch{epoch}.pt")
        
        scheduler.step()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Defactify Real Accuracy: {100*best_defactify_real:.2f}%")
    print("="*60)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train()
