#!/usr/bin/env python3
"""
Test the trained modern AI detector on Defactify test set.
"""

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ClassificationHead(nn.Module):
    """Same architecture as training."""
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.classifier(x)


def main():
    print("="*60)
    print("TESTING: Modern AI Detector (Your Trained Model)")
    print("="*60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load CLIP
    print("\nLoading CLIP backbone...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Load trained head
    print("Loading trained classification head...")
    head = ClassificationHead().to(device)
    weights_path = project_root / "models" / "modern_ai_detector.pt"
    head.load_state_dict(torch.load(weights_path, map_location=device))
    head.eval()
    print(f"Loaded weights from: {weights_path}")
    
    # Load test data
    print("\nLoading Defactify TEST set...")
    from datasets import load_dataset
    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")
    test_data = dataset["test"]
    print(f"Test set size: {len(test_data)}")
    
    # Sample for testing
    num_samples = 1000
    indices = random.sample(range(len(test_data)), min(num_samples, len(test_data)))
    
    real_scores = []
    fake_scores = []
    correct = 0
    total = 0
    
    print(f"\nTesting {len(indices)} images...")
    
    for idx in tqdm(indices, desc="Testing"):
        item = test_data[idx]
        image = item["Image"]
        label = item["Label_A"]  # 0=real, 1=fake
        
        if not isinstance(image, Image.Image):
            continue
            
        try:
            # Process
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            
            # Get prediction
            with torch.no_grad():
                emb = clip.get_image_features(pixel_values=pixel_values)
                logits = head(emb)
                probs = torch.softmax(logits, dim=-1)
                pred = logits.argmax(1).item()
                ai_prob = probs[0, 1].item()  # Probability of being AI
            
            # Track scores
            if label == 0:
                real_scores.append(ai_prob)
            else:
                fake_scores.append(ai_prob)
            
            # Track accuracy
            if pred == label:
                correct += 1
            total += 1
            
        except Exception as e:
            continue
    
    # Results
    accuracy = 100 * correct / total if total > 0 else 0
    real_mean = sum(real_scores) / len(real_scores) if real_scores else 0
    fake_mean = sum(fake_scores) / len(fake_scores) if fake_scores else 0
    separation = fake_mean - real_mean
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {len(real_scores)} real, {len(fake_scores)} fake")
    print(f"\nScore Distribution:")
    print(f"  Real Mean:  {real_mean:.3f} (should be LOW)")
    print(f"  Fake Mean:  {fake_mean:.3f} (should be HIGH)")
    print(f"  Separation: {separation:.3f}")
    print(f"\n*** ACCURACY: {accuracy:.1f}% ***")
    
    # Threshold analysis
    print(f"\nAccuracy at Different Thresholds:")
    print(f"  {'Threshold':<12} {'Real Acc':<12} {'Fake Acc':<12} {'Overall':<12}")
    print(f"  {'-'*48}")
    
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        real_correct = sum(1 for s in real_scores if s < thresh)
        fake_correct = sum(1 for s in fake_scores if s >= thresh)
        
        real_acc = 100 * real_correct / len(real_scores) if real_scores else 0
        fake_acc = 100 * fake_correct / len(fake_scores) if fake_scores else 0
        overall = 100 * (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
        
        marker = " ← default" if thresh == 0.5 else ""
        print(f"  {thresh:<12.2f} {real_acc:<12.1f}% {fake_acc:<12.1f}% {overall:<12.1f}%{marker}")
    
    print(f"\n{'='*60}")
    if accuracy >= 95:
        print("EXCELLENT! Model is production-ready!")
    elif accuracy >= 85:
        print("GOOD! Model performs well.")
    elif accuracy >= 70:
        print("MODERATE. Consider more training.")
    else:
        print("NEEDS IMPROVEMENT.")
    print(f"{'='*60}")


if __name__ == "__main__":
    random.seed(42)
    main()
