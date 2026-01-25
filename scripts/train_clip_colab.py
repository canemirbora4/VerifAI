"""
VerifAI - CLIP Head Training Script (Google Colab / Cloud GPU)
==============================================================

This script trains the CLIP ViT-L/14 classification head for AI detection.

Usage on Google Colab:
    1. Upload this script to Colab
    2. Change runtime to GPU (Runtime → Change runtime type → GPU)
    3. Run: !python train_clip_colab.py

The script will:
    - Download dataset from HuggingFace
    - Train the CLIP classification head
    - Save weights to clip_head_best.pt
"""

import os
import sys

# Install dependencies if needed
def install_deps():
    os.system("pip install -q torch torchvision transformers datasets pillow tqdm")

try:
    import torch
    from transformers import CLIPModel
except ImportError:
    print("Installing dependencies...")
    install_deps()
    import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "clip_model": "openai/clip-vit-large-patch14",
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "hidden_dims": (512, 256),
    "dropout": 0.3,
    "dataset": "prithivMLmods/AI-vs-Deepfake-vs-Real",  # Or use local data
}


# ============================================================================
# Model Definition
# ============================================================================

class ClassificationHead(nn.Module):
    """Trainable classification head for CLIP embeddings."""
    
    def __init__(self, input_dim=768, hidden_dims=(512, 256), num_classes=2, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.classifier(x)


class CLIPTrainer:
    """CLIP ViT-L/14 with frozen backbone + trainable head."""
    
    EMBEDDING_DIM = 768
    
    def __init__(self, model_name="openai/clip-vit-large-patch14", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Device: {self.device}")
        print(f"📦 Loading CLIP: {model_name}")
        
        # Load CLIP
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze CLIP
        self.clip_model.to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Classification head
        self.head = ClassificationHead(
            input_dim=self.EMBEDDING_DIM,
            hidden_dims=CONFIG["hidden_dims"],
            dropout=CONFIG["dropout"],
        ).to(self.device)
        
        clip_params = sum(p.numel() for p in self.clip_model.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"✅ CLIP params: {clip_params/1e6:.1f}M (frozen)")
        print(f"✅ Head params: {head_params/1e3:.1f}K (trainable)")
    
    def extract_embeddings(self, pixel_values):
        """Extract CLIP image embeddings."""
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(pixel_values=pixel_values)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features
    
    def train(self, train_loader, val_loader, epochs=10, lr=1e-3):
        """Train the classification head."""
        optimizer = torch.optim.AdamW(
            self.head.parameters(), 
            lr=lr, 
            weight_decay=CONFIG["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
        print(f"\n🚀 Starting training for {epochs} epochs...\n")
        
        for epoch in range(epochs):
            # Train
            self.head.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in pbar:
                labels = labels.to(self.device)
                
                embeddings = self.extract_embeddings(images)
                
                optimizer.zero_grad()
                logits = self.head(embeddings)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, pred = logits.max(1)
                train_total += labels.size(0)
                train_correct += pred.eq(labels).sum().item()
                
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    acc=f"{train_correct/train_total:.2%}"
                )
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validate
            val_loss, val_acc = self._evaluate(val_loader, criterion)
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            print(f"📊 Epoch {epoch+1}: train_acc={train_acc:.2%}, val_acc={val_acc:.2%}")
            
            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.head.state_dict(), "clip_head_best.pt")
                print(f"   💾 Saved best model (val_acc={val_acc:.2%})")
            
            scheduler.step()
        
        print(f"\n✅ Training complete!")
        print(f"🏆 Best validation accuracy: {best_acc:.2%}")
        
        return history
    
    def _evaluate(self, loader, criterion):
        self.head.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in loader:
                labels = labels.to(self.device)
                embeddings = self.extract_embeddings(images)
                logits = self.head(embeddings)
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                _, pred = logits.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        
        return total_loss / len(loader), correct / total
    
    def save_head(self, path):
        torch.save(self.head.state_dict(), path)
        print(f"💾 Saved head weights to {path}")


# ============================================================================
# Dataset
# ============================================================================

class HFDatasetWrapper(Dataset):
    """Wrapper for HuggingFace dataset."""
    
    def __init__(self, hf_dataset, transform=None, image_key="image", label_key="label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item[self.image_key]
        
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        label = item[self.label_key]
        # Map to binary: 0=Real, 1=AI (handles 3-class datasets too)
        if label == 2:  # Real in some datasets
            label = 0
        elif label in [0, 1]:  # AI/Deepfake
            label = 1
        
        return img, label


def load_dataset_hf():
    """Load dataset from HuggingFace."""
    from datasets import load_dataset
    
    print(f"📥 Loading dataset: {CONFIG['dataset']}")
    
    try:
        ds = load_dataset(CONFIG["dataset"], split="train")
        print(f"✅ Loaded {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("Trying fallback dataset...")
        
        # Fallback
        ds = load_dataset("birgermoell/ciFAKE", split="train")
        print(f"✅ Loaded CIFAKE: {len(ds)} samples")
        return ds


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("🎓 VerifAI - CLIP Head Training")
    print("=" * 60)
    print()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU available! Training will be slow.")
    print()
    
    # CLIP normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    
    # Load dataset
    hf_dataset = load_dataset_hf()
    full_dataset = HFDatasetWrapper(hf_dataset, transform=transform)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
    )
    
    # Initialize trainer
    trainer = CLIPTrainer(model_name=CONFIG["clip_model"])
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG["epochs"],
        lr=CONFIG["learning_rate"],
    )
    
    # Save final
    trainer.save_head("clip_head_final.pt")
    
    print()
    print("=" * 60)
    print("📁 Output files:")
    print("   - clip_head_best.pt  (best validation accuracy)")
    print("   - clip_head_final.pt (final epoch)")
    print()
    print("📋 Next steps:")
    print("   1. Download clip_head_best.pt")
    print("   2. Place in VerifAI/checkpoints/")
    print("   3. Use with: VerifAI(clip_head_path='checkpoints/clip_head_best.pt')")
    print("=" * 60)


if __name__ == "__main__":
    main()
