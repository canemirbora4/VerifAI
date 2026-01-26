"""
=============================================================================
COLAB TRAINING SCRIPT: Modern AI Image Detection
=============================================================================
Dataset: Defactify (DALL-E 3, Midjourney v6, SDXL, SD3 vs Real)
Model: CLIP ViT-L/14 (frozen) + Classification Head

INSTRUCTIONS:
1. Open Google Colab
2. Select GPU runtime (Runtime > Change runtime type > GPU)
3. Copy and paste this entire script into a cell
4. Run the cell
5. Download the trained weights when complete

=============================================================================
"""

# ============== INSTALL DEPENDENCIES ==============
# !pip install torch transformers datasets pillow tqdm -q

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os

# ============== CONFIGURATION ==============
CONFIG = {
    "clip_model": "openai/clip-vit-large-patch14",
    "batch_size": 32,  # Reduce if OOM
    "epochs": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "hidden_dims": (512, 256),
    "dropout": 0.3,
}

print("="*60)
print("MODERN AI IMAGE DETECTION TRAINING")
print("="*60)
print(f"Model: {CONFIG['clip_model']}")
print(f"Epochs: {CONFIG['epochs']}")
print(f"Batch size: {CONFIG['batch_size']}")
print("="*60)


# ============== CLASSIFICATION HEAD ==============
class ClassificationHead(nn.Module):
    """MLP classification head for CLIP embeddings."""
    
    def __init__(self, input_dim=768, hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary: Real vs AI
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


# ============== DATASET ==============
class DefactifyDataset(Dataset):
    """Dataset wrapper for Defactify."""
    
    def __init__(self, hf_dataset, processor):
        self.data = hf_dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["Image"]  # Capital I
        label = item["Label_A"]  # 0=real, 1=fake
        
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return pixel_values, label


# ============== SETUP ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

if device.type != "cuda":
    print("WARNING: No GPU detected! Training will be very slow.")
    print("Go to Runtime > Change runtime type > GPU")

# Load CLIP
print("\nLoading CLIP model...")
clip_model = CLIPModel.from_pretrained(CONFIG["clip_model"]).to(device)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False  # FREEZE backbone
print("CLIP backbone frozen ✓")

processor = CLIPProcessor.from_pretrained(CONFIG["clip_model"])

# Classification head
head = ClassificationHead(
    input_dim=768,
    hidden_dims=CONFIG["hidden_dims"],
    dropout=CONFIG["dropout"]
).to(device)
print(f"Classification head initialized ✓")

# Load dataset
print("\nLoading Defactify dataset...")
dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")
print(f"Train: {len(dataset['train'])} images")
print(f"Validation: {len(dataset['validation'])} images")
print(f"Test: {len(dataset['test'])} images")

# Create data loaders
train_dataset = DefactifyDataset(dataset["train"], processor)
val_dataset = DefactifyDataset(dataset["validation"], processor)

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG["batch_size"], 
    shuffle=True, 
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG["batch_size"], 
    num_workers=2,
    pin_memory=True
)

# Optimizer and loss
optimizer = torch.optim.AdamW(
    head.parameters(), 
    lr=CONFIG["learning_rate"], 
    weight_decay=CONFIG["weight_decay"]
)
criterion = nn.CrossEntropyLoss()

# ============== TRAINING ==============
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(CONFIG["epochs"]):
    # Training
    head.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
    for pixel_values, labels in pbar:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        # Extract CLIP embeddings
        with torch.no_grad():
            embeddings = clip_model.get_image_features(pixel_values=pixel_values)
        
        # Forward through head
        optimizer.zero_grad()
        outputs = head(embeddings)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Stats
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*train_correct/train_total:.1f}%"
        })
    
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    # Validation
    head.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for pixel_values, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]"):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            embeddings = clip_model.get_image_features(pixel_values=pixel_values)
            outputs = head(embeddings)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    # Save history
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    
    print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(head.state_dict(), "modern_ai_detector_best.pt")
        print(f"  ✓ New best model saved! (Val Acc: {val_acc:.1f}%)")

# Save final model
torch.save(head.state_dict(), "modern_ai_detector_final.pt")

# ============== RESULTS ==============
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
print(f"\nSaved files:")
print("  - modern_ai_detector_best.pt (best model)")
print("  - modern_ai_detector_final.pt (final model)")

# Download files
try:
    from google.colab import files
    print("\nDownloading best model...")
    files.download("modern_ai_detector_best.pt")
except:
    print("\nNot in Colab - files saved locally")

print("\n" + "="*60)
print("NEXT STEPS:")
print("1. Download 'modern_ai_detector_best.pt'")
print("2. Place it in your VerifAI/models/ folder")
print("3. Use it with CLIPDetector(head_weights_path='models/modern_ai_detector_best.pt')")
print("="*60)
