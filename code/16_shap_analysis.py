"""
16_shap_analysis.py
SHAP-based global interpretability analysis for the Stacking model.

Uses the Random Forest base learner inside the Stacking ensemble as the
surrogate model for SHAP TreeExplainer, computing SHAP values for class 1
(toxic) across the full descriptor + Morgan fingerprint feature space.

Input:  models/model_stacking.pkl
        data/step3_final_v4.csv
Output: results/shap_feature_importance.csv
        results/shap_descriptors.tif
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganGenerator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams.update({
    'font.family': 'Arial',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'font.size': 14,
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
})

print("=" * 70)
print("SHAP global interpretability analysis")
print("=" * 70)

# Load Stacking model package
with open(config.MODEL_STACKING, 'rb') as f:
    pkg = pickle.load(f)
model = pkg['model']
scaler = pkg['scaler']

# Sample 2000 compounds for SHAP computation (TreeExplainer is slow on the full set)
df = pd.read_csv(config.CSV_PATH)
df = df.sample(n=2000, random_state=config.RANDOM_STATE)
print(f"Sampled {len(df)} compounds for SHAP analysis")

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
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.NumAliphaticRings(mol),
            Descriptors.HeavyAtomCount(mol), Descriptors.FractionCSP3(mol),
        ])
        return np.concatenate([fp_array, descriptors])
    except Exception:
        return None


print("Computing features...")
X_list = []
for _, row in df.iterrows():
    feat = calc_features(row['smiles'])
    if feat is not None:
        X_list.append(feat)

X = np.array(X_list)
X[:, config.MORGAN_NBITS:] = scaler.transform(X[:, config.MORGAN_NBITS:])
print(f"Feature matrix: {X.shape}")

feature_names = [f'Morgan_{i}' for i in range(config.MORGAN_NBITS)] + [
    'MolWt', 'LogP', 'HBA', 'HBD', 'RotBonds',
    'TPSA', 'AromRings', 'AliphRings', 'HeavyAtoms', 'FracCSP3'
]

# Use the RF base learner inside the Stacking ensemble
print("\nComputing SHAP values (this may take a few minutes)...")
rf_model = model.named_estimators_['rf']
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)
print(f"shap_values shape: {shap_values.shape}")

shap_values_class1 = shap_values[:, :, 1]

# Global feature importance
mean_shap = np.abs(shap_values_class1).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_shap
}).sort_values('importance', ascending=False)

print("\nTop 20 important features:")
print(importance_df.head(20).to_string(index=False))
importance_df.to_csv(os.path.join(config.RESULTS_DIR, 'shap_feature_importance.csv'), index=False)

# Visualize: beeswarm plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_class1, X, feature_names=feature_names,
                  max_display=20, show=False)
plt.title('SHAP: Molecular Descriptors', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'shap_descriptors.tif'), dpi=600,
            bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print(f"\nSaved: {os.path.join(config.RESULTS_DIR, 'shap_descriptors.tif')}")