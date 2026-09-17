"""
03_knn_5fold.py
Train a K-Nearest Neighbors baseline model with 5-fold cross-validation.

Note: This version fixes the data leakage issue in the original code.
Standardization is performed independently within each fold.

Input:  data/step3_final_v4.csv
Output: Trained KNN model, predicted probabilities on the test set,
        and cross-validation AUC scores.
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

print("=" * 70)
print("Step 3: KNN training with 5-fold cross-validation")
print("=" * 70)

# ==================== Load data ====================
df = pd.read_csv(config.CSV_PATH)
print(f"Dataset loaded: {len(df)} compounds")
print(f"  Toxic:     {(df['label'] == 1).sum()} ({(df['label'] == 1).mean() * 100:.1f}%)")
print(f"  Non-toxic: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean() * 100:.1f}%)")

# ==================== Feature calculation ====================
_morgan_gen = GetMorganGenerator(radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_NBITS)

def calc_features(smiles):
    """Compute Morgan fingerprint (2048 bits) + 10 physicochemical descriptors."""
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        fp = _morgan_gen.GetFingerprint(mol)
        fp_array = np.array(fp)
        descriptors = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.FractionCSP3(mol),
        ])
        return np.concatenate([fp_array, descriptors])
    except Exception:
        return None


print("\nComputing molecular features...")
X_list, y_list = [], []
for idx, row in df.iterrows():
    feat = calc_features(row['smiles'])
    if feat is not None:
        X_list.append(feat)
        y_list.append(row['label'])
    if (idx + 1) % 1000 == 0:
        print(f"  Progress: {idx + 1}/{len(df)}")

X = np.array(X_list)
y = np.array(y_list)
print(f"Features computed: {len(X)} compounds, {X.shape[1]} dimensions")

# ==================== Train/Val/Test split ====================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE, stratify=y_temp
)

print(f"\nTraining set: {len(y_train)}")
print(f"Validation set: {len(y_val)}")
print(f"Test set: {len(y_test)}")

# ==================== 5-fold cross-validation ====================
# Standardization is fitted on each training fold only, then applied to
# both the training and validation portions of that fold.
print("\n" + "=" * 70)
print("KNN 5-fold cross-validation")
print("=" * 70)

cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

fold_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"\n--- Fold {fold + 1}/{config.N_FOLDS} ---")

    X_tr, X_val_fold = X_train[train_idx].copy(), X_train[val_idx].copy()
    y_tr, y_val_fold = y_train[train_idx], y_train[val_idx]
    print(f"  Train: {len(y_tr)}, Val: {len(y_val_fold)}")

    # Standardization inside the fold (no data leakage)
    scaler_fold = StandardScaler()
    X_tr[:, config.MORGAN_NBITS:] = scaler_fold.fit_transform(X_tr[:, config.MORGAN_NBITS:])
    X_val_fold[:, config.MORGAN_NBITS:] = scaler_fold.transform(X_val_fold[:, config.MORGAN_NBITS:])

    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_tr, y_tr)

    val_pred = knn.predict_proba(X_val_fold)[:, 1]
    auc = roc_auc_score(y_val_fold, val_pred)
    fold_scores.append(auc)
    elapsed = time.time() - start
    print(f"  Fold {fold + 1} AUC: {auc:.4f} | Time: {elapsed:.1f}s")

print(f"\n{'=' * 70}")
print(f"5-fold CV AUC: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
for i, score in enumerate(fold_scores):
    print(f"  Fold {i + 1}: {score:.4f}")

# ==================== Final model on full training set ====================
print(f"\n{'=' * 70}")
print("Training final KNN on full training set...")

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[:, config.MORGAN_NBITS:] = scaler.fit_transform(X_train[:, config.MORGAN_NBITS:])
X_val_scaled[:, config.MORGAN_NBITS:] = scaler.transform(X_val[:, config.MORGAN_NBITS:])
X_test_scaled[:, config.MORGAN_NBITS:] = scaler.transform(X_test[:, config.MORGAN_NBITS:])

knn_final = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_final.fit(X_train_scaled, y_train)

val_pred = knn_final.predict_proba(X_val_scaled)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
test_pred = knn_final.predict_proba(X_test_scaled)[:, 1]
test_auc = roc_auc_score(y_test, test_pred)
print(f"Validation AUC: {val_auc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# ==================== Save ====================
np.save(os.path.join(config.RESULTS_DIR, 'prob_knn.npy'), test_pred)
print("KNN prediction probabilities saved.")

model_package = {
    'model': knn_final,
    'scaler': scaler,
    'cv_auc': np.mean(fold_scores),
    'val_auc': val_auc,
}
with open(config.MODEL_KNN, 'wb') as f:
    pickle.dump(model_package, f)
print(f"\nKNN model saved to: {config.MODEL_KNN}")

print("\n" + "=" * 70)
print("KNN training complete.")
print("=" * 70)