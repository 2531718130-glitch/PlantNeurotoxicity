"""
05_nb_5fold.py
Train a Gaussian Naive Bayes baseline model with 5-fold cross-validation.

Note: Gaussian NB is used here as a baseline. It is expected to perform
poorly on the mixed feature space (binary fingerprint + continuous
descriptors) because it assumes feature independence and normality.
Its calibration failure and high prediction disagreement are analyzed
in the downstream ablation study.

Input:  data/step3_final_v4.csv
Output: Trained NB model and predicted probabilities on the test set.
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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

print("=" * 70)
print("Step 4: Naive Bayes training with 5-fold cross-validation")
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

# ==================== Standardization ====================
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[:, config.MORGAN_NBITS:] = scaler.fit_transform(X_train[:, config.MORGAN_NBITS:])
X_val_scaled[:, config.MORGAN_NBITS:] = scaler.transform(X_val[:, config.MORGAN_NBITS:])
X_test_scaled[:, config.MORGAN_NBITS:] = scaler.transform(X_test[:, config.MORGAN_NBITS:])

# ==================== 5-fold cross-validation ====================
print("\n" + "=" * 70)
print("Naive Bayes 5-fold cross-validation")
print("=" * 70)

cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

fold_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
    print(f"\n--- Fold {fold + 1}/{config.N_FOLDS} ---")
    X_tr, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val_fold = y_train[train_idx], y_train[val_idx]
    print(f"  Train: {len(y_tr)}, Val: {len(y_val_fold)}")

    start = time.time()
    nb = GaussianNB()
    nb.fit(X_tr, y_tr)

    val_pred = nb.predict_proba(X_val_fold)[:, 1]
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
print("Training final Naive Bayes on full training set...")

nb_final = GaussianNB()
nb_final.fit(X_train_scaled, y_train)

val_pred = nb_final.predict_proba(X_val_scaled)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.4f}")

# ==================== Save ====================
np.save(os.path.join(config.RESULTS_DIR, 'prob_nb.npy'),
        nb_final.predict_proba(X_test_scaled)[:, 1])
print("NB prediction probabilities saved.")

model_package = {
    'model': nb_final,
    'scaler': scaler,
    'cv_auc': np.mean(fold_scores),
    'val_auc': val_auc,
}
with open(config.MODEL_NB, 'wb') as f:
    pickle.dump(model_package, f)
print(f"\nNB model saved to: {config.MODEL_NB}")

print("\n" + "=" * 70)
print("Naive Bayes training complete.")
print("=" * 70)