"""
07_voting.py
Train a soft Voting ensemble (RF + XGBoost + SVM + KNN).
Collects out-of-fold (OOF) probabilities for all base learners and the ensemble.

Notes:
  - Standardization is performed inside each CV fold to avoid data leakage.
  - Naive Bayes is intentionally excluded, consistent with the Stacking model.

Input:  data/step3_final_v4.csv
Output: OOF probabilities and test-set predictions from the Voting ensemble.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganGenerator
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

print("=" * 70)
print("Step 6: Voting ensemble training with 5-fold cross-validation")
print("=" * 70)

# ==================== Load data ====================
df = pd.read_csv(config.CSV_PATH)
print(f"Dataset loaded: {len(df)} compounds")

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

# ==================== Helper: build base learners ====================
def build_base_learners(scale_pos_weight=1.0):
    """Return fresh instances of the four base learners used in Voting."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=config.RANDOM_STATE, n_jobs=-1
    )
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', random_state=config.RANDOM_STATE, n_jobs=-1
    )
    svm = SVC(probability=True, class_weight='balanced', kernel='rbf',
              C=1.0, random_state=config.RANDOM_STATE)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    return [('rf', rf), ('xgb', xgb_m), ('svm', svm), ('knn', knn)]


# ==================== 5-fold cross-validation with OOF collection ====================
print("\n" + "=" * 70)
print("Voting 5-fold cross-validation (in-fold standardization, OOF collected)")
print("=" * 70)

skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

model_names = ['KNN', 'Random Forest', 'XGBoost', 'SVM', 'Voting']
oof_probs = {name: np.zeros(len(y)) for name in model_names}
oof_labels = np.zeros(len(y))
cv_aucs = {name: [] for name in model_names}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{config.N_FOLDS} ---")
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Standardization inside the fold
    scaler_cv = StandardScaler()
    X_tr_s = X_tr.copy()
    X_val_s = X_val.copy()
    X_tr_s[:, config.MORGAN_NBITS:] = scaler_cv.fit_transform(X_tr[:, config.MORGAN_NBITS:])
    X_val_s[:, config.MORGAN_NBITS:] = scaler_cv.transform(X_val[:, config.MORGAN_NBITS:])

    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    estimators = build_base_learners(scale_pos_weight)
    voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

    models = {
        'KNN': estimators[3][1],
        'Random Forest': estimators[0][1],
        'XGBoost': estimators[1][1],
        'SVM': estimators[2][1],
        'Voting': voting,
    }

    for name, model in models.items():
        model.fit(X_tr_s, y_tr)
        prob = model.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, prob)
        cv_aucs[name].append(auc)
        oof_probs[name][val_idx] = prob
        print(f"  {name:<16} AUC: {auc:.4f}")

    oof_labels[val_idx] = y_val

# ==================== Summary ====================
print("\n" + "=" * 70)
print("5-fold CV AUC summary (mean +/- SD)")
print("=" * 70)
print(f"{'Model':<18} {'5-CV AUC':>20}")
print("-" * 45)
for name in model_names:
    mean_auc = np.mean(cv_aucs[name])
    std_auc = np.std(cv_aucs[name])
    print(f"{name:<18} {mean_auc:.4f} +/- {std_auc:.4f}")

# ==================== Save OOF probabilities ====================
for name, probs in oof_probs.items():
    key = name.lower().replace(' ', '_')
    np.save(os.path.join(config.RESULTS_DIR, f'voting_prob_{key}_oof.npy'), probs)
np.save(os.path.join(config.RESULTS_DIR, 'voting_label_oof.npy'), oof_labels)
print(f"\nOOF probabilities saved to: {config.RESULTS_DIR}")

# ==================== Final model on fixed split ====================
print("\n" + "=" * 70)
print("Final Voting model trained on fixed train/val/test split")
print("=" * 70)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE, stratify=y_temp
)

print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

scaler = StandardScaler()
X_train_s = X_train.copy()
X_val_s = X_val.copy()
X_test_s = X_test.copy()
X_train_s[:, config.MORGAN_NBITS:] = scaler.fit_transform(X_train[:, config.MORGAN_NBITS:])
X_val_s[:, config.MORGAN_NBITS:] = scaler.transform(X_val[:, config.MORGAN_NBITS:])
X_test_s[:, config.MORGAN_NBITS:] = scaler.transform(X_test[:, config.MORGAN_NBITS:])

scale_pos_weight_fix = (y_train == 0).sum() / (y_train == 1).sum()
estimators_f = build_base_learners(scale_pos_weight_fix)
voting_f = VotingClassifier(estimators=estimators_f, voting='soft', n_jobs=-1)
voting_f.fit(X_train_s, y_train)

val_auc = roc_auc_score(y_val, voting_f.predict_proba(X_val_s)[:, 1])
test_auc = roc_auc_score(y_test, voting_f.predict_proba(X_test_s)[:, 1])
print(f"\nVoting validation AUC: {val_auc:.4f}")
print(f"Voting test AUC: {test_auc:.4f}")

# ==================== Save ====================
with open(config.MODEL_VOTING, 'wb') as f:
    pickle.dump({'model': voting_f, 'scaler': scaler}, f)

np.save(os.path.join(config.RESULTS_DIR, 'voting_prob_test.npy'),
        voting_f.predict_proba(X_test_s)[:, 1])
np.save(os.path.join(config.RESULTS_DIR, 'voting_label_test.npy'), y_test)

print(f"\n{'=' * 70}")
print("Voting model saved.")
print("OOF probabilities saved.")
print(f"{'=' * 70}")