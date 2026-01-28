"""
Frequency Classifier Training with sklearn
===========================================
Uses FrequencyExtractor to extract FFT/DCT features,
then trains XGBoost/RandomForest classifier.

No GPU required - runs on CPU.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from tqdm import tqdm
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from verifai.features.frequency import FrequencyExtractor

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_NAME = "Rajarshi-Roy-research/Defactify_Image_Dataset"
NUM_SAMPLES = 2000  # Total samples to use (1000 real + 1000 fake ideally)
RANDOM_STATE = 42

print("="*60)
print("Frequency Classifier Training (sklearn)")
print("="*60)

# ============================================================
# 1. Initialize Feature Extractor
# ============================================================
print("\n[1/5] Initializing FrequencyExtractor...")
extractor = FrequencyExtractor(
    image_size=(256, 256),
    patch_size=64,
    num_azimuthal_bins=64,
    compute_patches=True,
    normalize=True,
)

# Get feature dimension
feature_dim = extractor.get_feature_dim()
print(f"Feature dimension: {feature_dim}")

# ============================================================
# 2. Load Dataset
# ============================================================
print(f"\n[2/5] Loading dataset: {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME)

# Use train split for training
train_data = dataset['train']
print(f"Total training samples available: {len(train_data)}")

# Sample balanced data
print(f"Sampling {NUM_SAMPLES} images (balanced)...")

# Separate by label
real_indices = [i for i in range(len(train_data)) if train_data[i]['Label_A'] == 0]
fake_indices = [i for i in range(len(train_data)) if train_data[i]['Label_A'] == 1]

print(f"Available: {len(real_indices)} real, {len(fake_indices)} fake")

# Balance sampling
np.random.seed(RANDOM_STATE)
n_per_class = min(NUM_SAMPLES // 2, len(real_indices), len(fake_indices))
sampled_real = np.random.choice(real_indices, n_per_class, replace=False)
sampled_fake = np.random.choice(fake_indices, n_per_class, replace=False)
sampled_indices = list(sampled_real) + list(sampled_fake)
np.random.shuffle(sampled_indices)

print(f"Using {n_per_class} real + {n_per_class} fake = {len(sampled_indices)} total")

# ============================================================
# 3. Extract Features
# ============================================================
print(f"\n[3/5] Extracting frequency features...")

features_list = []
labels_list = []
errors = 0

for idx in tqdm(sampled_indices, desc="Extracting"):
    try:
        item = train_data[idx]
        image = item['Image'].convert('RGB')
        label = item['Label_A']
        
        # Extract features
        freq_features = extractor.extract(image)
        feature_vector = freq_features.feature_vector
        
        if feature_vector is not None and len(feature_vector) == feature_dim:
            features_list.append(feature_vector)
            labels_list.append(label)
    except Exception as e:
        errors += 1
        continue

X = np.array(features_list)
y = np.array(labels_list)

print(f"Extracted features: {X.shape}")
print(f"Labels distribution: {np.bincount(y)}")
if errors > 0:
    print(f"Errors: {errors}")

# ============================================================
# 4. Train Classifiers
# ============================================================
print(f"\n[4/5] Training classifiers...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Train Random Forest
print("\n--- Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc*100:.1f}%")

# Per-class accuracy
rf_real_acc = accuracy_score(y_test[y_test==0], rf_pred[y_test==0])
rf_fake_acc = accuracy_score(y_test[y_test==1], rf_pred[y_test==1])
print(f"  Real Accuracy: {rf_real_acc*100:.1f}%")
print(f"  Fake Accuracy: {rf_fake_acc*100:.1f}%")

# Train Gradient Boosting (XGBoost-like)
print("\n--- Gradient Boosting ---")
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
)
gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"Gradient Boosting Accuracy: {gb_acc*100:.1f}%")

# Per-class accuracy
gb_real_acc = accuracy_score(y_test[y_test==0], gb_pred[y_test==0])
gb_fake_acc = accuracy_score(y_test[y_test==1], gb_pred[y_test==1])
print(f"  Real Accuracy: {gb_real_acc*100:.1f}%")
print(f"  Fake Accuracy: {gb_fake_acc*100:.1f}%")

# ============================================================
# 5. Save Best Model
# ============================================================
print(f"\n[5/5] Saving models...")

# Choose best model
if rf_acc >= gb_acc:
    best_model = rf
    best_name = "RandomForest"
    best_acc = rf_acc
else:
    best_model = gb
    best_name = "GradientBoosting"
    best_acc = gb_acc

# Save both models
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(models_dir, exist_ok=True)

rf_path = os.path.join(models_dir, 'frequency_classifier_rf.joblib')
gb_path = os.path.join(models_dir, 'frequency_classifier_gb.joblib')
best_path = os.path.join(models_dir, 'frequency_classifier.joblib')

joblib.dump(rf, rf_path)
joblib.dump(gb, gb_path)
joblib.dump(best_model, best_path)

print(f"Saved: {rf_path}")
print(f"Saved: {gb_path}")
print(f"Saved: {best_path} ({best_name})")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best Model: {best_name}")
print(f"Accuracy: {best_acc*100:.1f}%")
print(f"Feature Dimension: {feature_dim}")
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")
print("="*60)

# Detailed classification report for best model
print("\nClassification Report (Best Model):")
if best_name == "RandomForest":
    print(classification_report(y_test, rf_pred, target_names=['Real', 'AI-Generated']))
else:
    print(classification_report(y_test, gb_pred, target_names=['Real', 'AI-Generated']))

print("\nConfusion Matrix:")
if best_name == "RandomForest":
    print(confusion_matrix(y_test, rf_pred))
else:
    print(confusion_matrix(y_test, gb_pred))
