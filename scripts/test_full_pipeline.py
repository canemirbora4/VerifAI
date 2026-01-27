"""
Test Full VerifAI Pipeline
===========================
CLIP (trained) + PRNU + Metadata ensemble
(Frequency KAPALI - train edilmedi)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import tempfile

# Import VerifAI pipeline
from verifai.pipeline import VerifAI


def main():
    print("="*60)
    print("FULL VERIFAI PIPELINE TEST")
    print("="*60)
    
    # Paths
    clip_head_path = project_root / "models" / "modern_ai_detector.pt"
    
    print(f"\nCLIP head weights: {clip_head_path}")
    print(f"Exists: {clip_head_path.exists()}")
    
    # Initialize VerifAI with our trained CLIP head
    # NOT using frequency - it's untrained and gives bad results
    print("\nInitializing VerifAI pipeline...")
    detector = VerifAI(
        use_clip=True,
        clip_head_path=str(clip_head_path),
        use_frequency=False,  # KAPALI - train edilmedi
        use_metadata=True,
        use_prnu=True,
        use_provenance=False,  # C2PA genelde yok
        fusion_method="weighted",
        threshold=0.5,
    )
    
    # Load models
    print("Loading models...")
    detector.load()
    
    # Print pipeline info
    info = detector.get_info()
    print(f"\nPipeline config:")
    print(f"  Detectors: {info['detectors']}")
    print(f"  Weights: {info['ensemble']['weights']}")
    print(f"  Fusion: {info['ensemble']['method']}")
    
    # Load test dataset
    print("\nLoading Defactify TEST set...")
    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")
    test_data = dataset["test"]
    print(f"Test set size: {len(test_data)}")
    
    # Sample
    n_samples = 200  # Daha az sample çünkü pipeline yavaş olabilir
    indices = random.sample(range(len(test_data)), min(n_samples, len(test_data)))
    
    # Track results
    results = {
        "correct": 0,
        "total": 0,
        "real_correct": 0,
        "real_total": 0,
        "fake_correct": 0,
        "fake_total": 0,
        "scores": [],
        "detector_scores": {"neural": [], "metadata": [], "prnu": []}
    }
    
    print(f"\nTesting {n_samples} images...")
    
    for idx in tqdm(indices, desc="Testing"):
        item = test_data[idx]
        image = item["Image"].convert("RGB")
        label = item["Label_A"]  # 0=real, 1=fake
        
        try:
            # Save image temporarily (pipeline needs file for metadata/PRNU)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                temp_path = f.name
                image.save(temp_path, "JPEG", quality=95)
            
            # Run full pipeline
            result = detector.detect(temp_path, return_evidence=False)
            
            # Get prediction
            ai_prob = result.confidence
            pred = 1 if result.is_ai_generated else 0
            
            # Track results
            results["total"] += 1
            results["correct"] += int(pred == label)
            results["scores"].append((ai_prob, label))
            
            # Track per-detector scores
            for det_name, det_score in result.detector_scores.items():
                if det_name in results["detector_scores"]:
                    results["detector_scores"][det_name].append((det_score, label))
            
            if label == 0:  # Real
                results["real_total"] += 1
                results["real_correct"] += int(pred == label)
            else:  # Fake
                results["fake_total"] += 1
                results["fake_correct"] += int(pred == label)
            
            # Cleanup
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"\nError processing image {idx}: {e}")
            continue
    
    # Calculate metrics
    def calc_metrics(scores_list):
        if not scores_list:
            return {"mean": 0, "separation": 0}
        real_scores = [s for s, l in scores_list if l == 0]
        fake_scores = [s for s, l in scores_list if l == 1]
        real_mean = np.mean(real_scores) if real_scores else 0
        fake_mean = np.mean(fake_scores) if fake_scores else 0
        return {
            "real_mean": real_mean,
            "fake_mean": fake_mean,
            "separation": fake_mean - real_mean
        }
    
    overall_metrics = calc_metrics(results["scores"])
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS - FULL PIPELINE")
    print("="*60)
    print(f"Samples: {results['real_total']} real, {results['fake_total']} fake")
    
    accuracy = 100 * results["correct"] / results["total"] if results["total"] > 0 else 0
    real_acc = 100 * results["real_correct"] / results["real_total"] if results["real_total"] > 0 else 0
    fake_acc = 100 * results["fake_correct"] / results["fake_total"] if results["fake_total"] > 0 else 0
    
    print(f"\n*** OVERALL ACCURACY: {accuracy:.1f}% ***")
    print(f"    Real Accuracy:   {real_acc:.1f}%")
    print(f"    Fake Accuracy:   {fake_acc:.1f}%")
    print(f"    Separation:      {overall_metrics['separation']:.3f}")
    print(f"    Real Mean:       {overall_metrics['real_mean']:.3f}")
    print(f"    Fake Mean:       {overall_metrics['fake_mean']:.3f}")
    
    # Per-detector breakdown
    print("\n" + "-"*60)
    print("PER-DETECTOR SCORES")
    print("-"*60)
    
    for det_name, det_scores in results["detector_scores"].items():
        if det_scores:
            det_metrics = calc_metrics(det_scores)
            print(f"\n{det_name.upper()}:")
            print(f"  Real Mean:   {det_metrics['real_mean']:.3f}")
            print(f"  Fake Mean:   {det_metrics['fake_mean']:.3f}")
            print(f"  Separation:  {det_metrics['separation']:.3f}")
    
    print("\n" + "="*60)
    if accuracy >= 85:
        print("EXCELLENT! Full pipeline working well.")
    elif accuracy >= 80:
        print("GOOD. Pipeline shows improvement.")
    else:
        print("MODERATE. May need tuning.")
    print("="*60)


if __name__ == "__main__":
    random.seed(42)
    main()
