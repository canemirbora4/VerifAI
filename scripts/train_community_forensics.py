# ============================================================
# CommunityForensics Training Script - SHARD VERSION
# Downloads only a fraction of the dataset to stay within limits
# Keeps full random access and all features
# ============================================================

import os
import io
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
SEED = 42
SAVE_EVERY = 1
VAL_EVERY = 1

# SHARD CONFIGURATION - Only download part of the dataset
# CommunityForensics-Small has ~356K rows
# We'll use 10% = ~35K samples (fits in Kaggle disk)
NUM_SHARDS = 10          # Split dataset into 10 parts
SHARD_INDEX = 0          # Use first shard (change to 0-9 for different data)

# Anti-shortcut
JPEG_QUALITY_MIN = 70
JPEG_QUALITY_MAX = 100

# Generator-aware split
VAL_GENERATOR_RATIO = 0.10

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
# DATASET WITH SHARD LOADING
# ============================================================
class CommunityForensicsDataset(Dataset):
    """
    Loads only a shard of the dataset to stay within disk limits.
    Keeps random access and all features.
    """
    def __init__(self, processor, split='train', val_generators=None, enforce_balance=None):
        print(f"Loading CommunityForensics-Small SHARD {SHARD_INDEX}/{NUM_SHARDS} ({split})...")
        
        # Load only one shard - much smaller download
        self.ds = load_dataset(
            "OwensLab/CommunityForensics-Small",
            split=f"train[{SHARD_INDEX}%:{SHARD_INDEX + (100//NUM_SHARDS)}%]",
            trust_remote_code=True
        )
        
        print(f"Shard loaded: {len(self.ds)} samples")
        
        self.processor = processor
        self.split = split
        self.is_train = (split == 'train')
        
        # Deterministic RNG for reproducible splits
        rng = random.Random(SEED)
        
        # Fast column-based indexing
        print("Indexing metadata...")
        labels_raw = self.ds["label"]
        model_names_raw = self.ds["model_name"]
        
        labels = [int(x) if x is not None else 1 for x in labels_raw]
        model_names = [x if x is not None else "unknown" for x in model_names_raw]
        
        # Build indices
        real_indices = [i for i, y in enumerate(labels) if y == 0]
        fake_by_gen = defaultdict(list)
        for i, y in enumerate(labels):
            if y == 1:
                g = model_names[i] or "unknown"
                fake_by_gen[g].append(i)
        
        all_generators = list(fake_by_gen.keys())
        total_fake = sum(len(v) for v in fake_by_gen.values())
        
        print(f"Found {len(real_indices)} real, {total_fake} fake, {len(all_generators)} generators")
        
        # Shuffle real indices
        rng.shuffle(real_indices)
        
        # Generator-aware split
        if val_generators is None:
            rng.shuffle(all_generators)
            n_val = max(1, int(len(all_generators) * VAL_GENERATOR_RATIO))
            self.val_generators = set(all_generators[:n_val])
            print(f"Held out {len(self.val_generators)} generators for validation")
        else:
            self.val_generators = val_generators
        
        # Build split indices
        if split == 'train':
            split_point = int(len(real_indices) * 0.9)
            self.real_indices = real_indices[:split_point]
            self.fake_indices = []
            for gen, idxs in fake_by_gen.items():
                if gen not in self.val_generators:
                    self.fake_indices.extend(idxs)
        else:
            split_point = int(len(real_indices) * 0.9)
            self.real_indices = real_indices[split_point:]
            self.fake_indices = []
            for gen in self.val_generators:
                self.fake_indices.extend(fake_by_gen.get(gen, []))
        
        print(f"{split}: {len(self.real_indices)} real + {len(self.fake_indices)} fake")
        
        # Balance for val
        if enforce_balance is None:
            enforce_balance = (split == 'val')
        
        if enforce_balance:
            min_class = min(len(self.real_indices), len(self.fake_indices))
            if min_class > 0:
                rng.shuffle(self.real_indices)
                rng.shuffle(self.fake_indices)
                self.real_indices = self.real_indices[:min_class]
                self.fake_indices = self.fake_indices[:min_class]
        
        # Build cache and indices
        self.labels_cache = {idx: 0 for idx in self.real_indices}
        self.labels_cache.update({idx: 1 for idx in self.fake_indices})
        
        self.indices = self.real_indices + self.fake_indices
        rng.shuffle(self.indices)
        
        print(f"Final {split}: {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                sample = self.ds[real_idx]
                image = decode_image(sample)
                label = self.labels_cache.get(real_idx, int(sample['label']))
                image = anti_shortcut_preprocess(image, is_train=self.is_train)
                inputs = self.processor(images=image, return_tensors='pt')
                return inputs['pixel_values'].squeeze(0), label
            except Exception as e:
                if retry < max_retries - 1:
                    real_idx = random.choice(self.indices)
                else:
                    fallback_idx = random.choice(self.indices)
                    sample = self.ds[fallback_idx]
                    image = decode_image(sample)
                    image = anti_shortcut_preprocess(image, is_train=self.is_train)
                    inputs = self.processor(images=image, return_tensors='pt')
                    return inputs['pixel_values'].squeeze(0), self.labels_cache[fallback_idx]

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
    raise RuntimeError("Could not locate image_embeds")

def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============================================================
# EXTERNAL VALIDATION (Defactify)
# ============================================================
def evaluate_on_defactify(model, head, processor, device, n_samples=50):
    print("\n  Evaluating on Defactify...")
    try:
        ds = load_dataset(
            "Rajarshi-Roy-research/Defactify_Image_Dataset",
            split="test",
            streaming=True
        )
    except:
        return 0, 0, 0
    
    model.eval()
    head.eval()
    real_correct, real_total, ai_correct, ai_total = 0, 0, 0, 0
    
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
                features = get_clip_features(model, inputs['pixel_values'].to(device))
                pred = torch.argmax(head(features), dim=1).item()
                
                if true_label == 0:
                    real_total += 1
                    real_correct += (pred == 0)
                else:
                    ai_total += 1
                    ai_correct += (pred == 1)
            except:
                continue
    
    return real_correct/max(1,real_total), ai_correct/max(1,ai_total), (real_correct+ai_correct)/max(1,real_total+ai_total)

# ============================================================
# TRAINING LOOP
# ============================================================
def train():
    print("="*60)
    print(f"CommunityForensics Training - SHARD {SHARD_INDEX}/{NUM_SHARDS}")
    print("="*60)
    
    # Load CLIP
    print("\nLoading CLIP...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    for param in model.parameters():
        param.requires_grad = False
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device).eval()
    
    head = ClassificationHead()
    if num_gpus > 1:
        head = nn.DataParallel(head)
    head = head.to(device)
    
    # Datasets
    train_dataset = CommunityForensicsDataset(processor, 'train', enforce_balance=False)
    val_dataset = CommunityForensicsDataset(processor, 'val', train_dataset.val_generators, enforce_balance=True)
    
    # Sampler for balance
    train_labels = [train_dataset.labels_cache[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=2)
    weights = [1.0 / class_counts[y] for y in train_labels]
    sampler = WeightedRandomSampler(weights, len(train_labels), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE*max(1,num_gpus), 
                              sampler=sampler, num_workers=NUM_WORKERS, 
                              pin_memory=True, drop_last=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*max(1,num_gpus),
                            num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker)
    
    # Optimizer
    head_params = head.module.parameters() if hasattr(head, 'module') else head.parameters()
    optimizer = torch.optim.AdamW(head_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    best_defactify_real = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{EPOCHS}\n{'='*60}")
        
        head.train()
        train_correct, train_total = 0, 0
        
        for pixel_values, labels in tqdm(train_loader, desc="Training"):
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                with torch.no_grad():
                    features = get_clip_features(model, pixel_values)
                loss = criterion(head(features).float(), labels.long())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_correct += (torch.argmax(head(features), 1) == labels).sum().item()
            train_total += labels.size(0)
        
        print(f"Train Acc: {100*train_correct/train_total:.2f}%")
        
        # Validation
        head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for pixel_values, labels in tqdm(val_loader, desc="Validation"):
                pixel_values, labels = pixel_values.to(device), labels.to(device)
                features = get_clip_features(model, pixel_values)
                val_correct += (torch.argmax(head(features), 1) == labels).sum().item()
                val_total += labels.size(0)
        
        print(f"Val Acc: {100*val_correct/val_total:.2f}%")
        
        # Defactify test
        def_real, def_ai, _ = evaluate_on_defactify(model, head, processor, device)
        print(f"Defactify: Real={100*def_real:.1f}%, AI={100*def_ai:.1f}%")
        
        if def_real > best_defactify_real:
            best_defactify_real = def_real
            save_head = head.module if hasattr(head, 'module') else head
            torch.save(save_head.state_dict(), 'cf_detector_best.pt')
            print(f"  ★ Best model saved!")
        
        save_head = head.module if hasattr(head, 'module') else head
        torch.save(save_head.state_dict(), f'cf_detector_epoch{epoch}.pt')
        
        scheduler.step()
    
    print(f"\nTraining Complete! Best Defactify Real: {100*best_defactify_real:.2f}%")

if __name__ == "__main__":
    train()
