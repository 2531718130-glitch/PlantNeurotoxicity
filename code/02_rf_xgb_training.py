"""
02_rf_xgb_training.py
Train Random Forest and XGBoost baseline models.

Input:  data/step3_final_v4.csv (SMILES + label)
Output: Trained RF and XGBoost models, plus predicted probabilities on the test set.
        Results saved to results/ directory.
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path so that config can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

print("=" * 70)
print("Step 1: Baseline model training (Random Forest + XGBoost)")
print("=" * 70)

# ==================== Load data ====================
df = pd.read_csv(config.CSV_PATH)
print(f"Dataset loaded: {len(df)} compounds")
print(f"  Toxic:   {(df['label'] == 1).sum()} ({(df['label'] == 1).mean() * 100:.1f}%)")
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

# ==================== Feature standardization ====================
# Only the 10 continuous descriptors (columns 2048 onward) are standardized;
# Morgan fingerprint bits are binary and are left unchanged.
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[:, config.MORGAN_NBITS:] = scaler.fit_transform(X_train[:, config.MORGAN_NBITS:])
X_val_scaled[:, config.MORGAN_NBITS:] = scaler.transform(X_val[:, config.MORGAN_NBITS:])
X_test_scaled[:, config.MORGAN_NBITS:] = scaler.transform(X_test[:, config.MORGAN_NBITS:])

# ==================== Class weight ====================
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\nClass ratio (neg:pos) = {neg_count}:{pos_count} (scale_pos_weight = {scale_pos_weight:.2f})")

# ==================== Baseline model training ====================
print("\n" + "=" * 70)
print("Baseline model training (5-fold cross-validation)")
print("=" * 70)

cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

# ----- Random Forest -----
print("\n--- Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=5,
    class_weight='balanced', random_state=config.RANDOM_STATE, n_jobs=-1
)
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
rf.fit(X_train_scaled, y_train)
val_pred = rf.predict_proba(X_val_scaled)[:, 1]
print(f"  5-fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Validation AUC: {roc_auc_score(y_val, val_pred):.4f}")

# ----- XGBoost -----
print("\n--- XGBoost ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss', random_state=config.RANDOM_STATE, n_jobs=-1
)
cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
xgb_model.fit(X_train_scaled, y_train)
val_pred = xgb_model.predict_proba(X_val_scaled)[:, 1]
print(f"  5-fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Validation AUC: {roc_auc_score(y_val, val_pred):.4f}")

# ==================== Save predictions ====================
np.save(os.path.join(config.RESULTS_DIR, 'y_true.npy'), y_test)
np.save(os.path.join(config.RESULTS_DIR, 'prob_rf.npy'), rf.predict_proba(X_test_scaled)[:, 1])
np.save(os.path.join(config.RESULTS_DIR, 'prob_xgb.npy'), xgb_model.predict_proba(X_test_scaled)[:, 1])
print("\nRF / XGBoost prediction probabilities saved.")

# ==================== Save models ====================
with open(config.MODEL_RF_XGB, 'wb') as f:
    pickle.dump({'rf': rf, 'xgb': xgb_model, 'scaler': scaler}, f)
print(f"Models saved to: {config.MODEL_RF_XGB}")

print("\n" + "=" * 70)
print("Training complete.")
print("=" * 70)