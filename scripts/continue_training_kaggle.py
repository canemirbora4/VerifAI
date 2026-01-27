# KAGGLE'DA ÇALIŞTIR - Epoch 5'ten devam + Weighted Loss
# 1. Önce epoch5 .pt dosyasını Kaggle'a yükle
# 2. Bu kodu çalıştır

!pip install torch transformers datasets pillow tqdm -q

import torch
import torch.nn as nn
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
EPOCHS = 10  # 5 epoch daha (6-15 arası)
LR = 5e-4    # Biraz düşük LR (fine-tuning için)
START_EPOCH = 6  # Epoch 5'ten devam

# Weighted Loss - Real sınıfına daha fazla ağırlık
# Dataset: ~15% real, ~85% fake -> Real'e 5x ağırlık
REAL_WEIGHT = 5.0
FAKE_WEIGHT = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Config: LR={LR}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
print(f"Weighted Loss: Real={REAL_WEIGHT}, Fake={FAKE_WEIGHT}")

# ============================================================
# MODEL
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
        label = item["Label_A"]  # 0=real, 1=fake
        return inputs["pixel_values"].squeeze(0), label

# ============================================================
# LOAD MODELS
# ============================================================
print("\n" + "="*60)
print("Loading CLIP backbone...")
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip.eval()
for p in clip.parameters():
    p.requires_grad = False

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

print("Loading classification head from epoch 5...")
head = ClassificationHead().to(device)

# Epoch 5 ağırlıklarını yükle
# Kaggle'a yüklediğin dosya adını buraya yaz
WEIGHTS_PATH = "modern_ai_detector.pt"  # veya "/kaggle/input/xxx/modern_ai_detector.pt"
head.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
print(f"✓ Loaded weights from: {WEIGHTS_PATH}")

# ============================================================
# LOAD DATASET
# ============================================================
print("\nLoading Defactify dataset...")
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
optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Weighted Loss - Real sınıfına daha fazla ağırlık ver
weights = torch.tensor([REAL_WEIGHT, FAKE_WEIGHT]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
print(f"✓ Using weighted loss: {weights.tolist()}")

best_acc = 0.82  # Epoch 5'in accuracy'si

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "="*60)
print("STARTING TRAINING FROM EPOCH 6")
print("="*60)

for epoch in range(EPOCHS):
    actual_epoch = START_EPOCH + epoch
    head.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {actual_epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            features = clip.get_image_features(pixel_values=images)
        
        logits = head(features)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.1f}%'
        })
    
    scheduler.step()
    train_acc = 100 * correct / total
    
    # Validation
    head.eval()
    val_correct = 0
    val_total = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            features = clip.get_image_features(pixel_values=images)
            logits = head(features)
            _, predicted = torch.max(logits, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            # Real/Fake ayrı ayrı
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
    print(f"EPOCH {actual_epoch} COMPLETE")
    print(f"{'='*60}")
    print(f"Train Acc: {train_acc:.1f}%")
    print(f"Val Acc:   {val_acc:.1f}%")
    print(f"  - Real Acc:  {real_acc:.1f}%")
    print(f"  - Fake Acc:  {fake_acc:.1f}%")
    print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Her epoch kaydet
    save_name = f"modern_ai_detector_epoch{actual_epoch}.pt"
    torch.save(head.state_dict(), save_name)
    print(f"✓ Saved: {save_name}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(head.state_dict(), "modern_ai_detector_best.pt")
        print(f"★ NEW BEST! Saved as modern_ai_detector_best.pt")

print("\n" + "="*60)
print(f"TRAINING COMPLETE! Best accuracy: {best_acc:.1f}%")
print("="*60)

# Download link
from IPython.display import FileLink
print("\nDownload best model:")
display(FileLink("modern_ai_detector_best.pt"))
