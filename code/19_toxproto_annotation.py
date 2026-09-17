"""
19_toxproto_annotation.py
Annotate ToxProto prototypes by substructure enrichment (Fisher's exact test).

Input:  results/dtrl/toxproto_results.npz
        data/step3_final_v4.csv
Output: Console report of significantly enriched substructures per prototype.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import fisher_exact

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Load ToxProto results
toxproto_path = os.path.join(config.RESULTS_DIR, 'dtrl', 'toxproto_results.npz')
data = np.load(toxproto_path)
cluster_labels = data['cluster_labels']
labels = data['labels']

df = pd.read_csv(config.CSV_PATH)
smiles_list = df['smiles'].tolist()[:len(cluster_labels)]

# Substructure patterns
patterns = {
    'pyrido_pyrimidine': 'n1cnc2[c,n]cccc2[c,n]1',
    'indole': 'c1ccc2[nH]ccc2c1',
    'pyrimidine': '[n,c]1[n,c][n,c][n,c][n,c]1',
    'sulfonamide': '[S](=O)(=O)[N,n]',
    'nitro': '[N+](=O)[O-]',
    'quinoline': 'c1ccc2ncccc2c1',
    'pyrazole': 'n1[nH]ccc1',
    'cyano': 'C#N',
    'sulfonyl': '[S](=O)(=O)',
}

print("=" * 70)
print("ToxProto prototype substructure enrichment")
print("=" * 70)

for k in range(6):
    mask = cluster_labels == k
    proto_smiles = [s for i, s in enumerate(smiles_list) if mask[i]]
    bg_smiles = [s for i, s in enumerate(smiles_list) if not mask[i]]

    print(f"\nPrototype {k} ({mask.sum()} molecules):")
    for name, smarts in patterns.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue

        proto_hits = sum(1 for smi in proto_smiles
                         if Chem.MolFromSmiles(smi) and Chem.MolFromSmiles(smi).HasSubstructMatch(patt))
        bg_hits = sum(1 for smi in bg_smiles
                      if Chem.MolFromSmiles(smi) and Chem.MolFromSmiles(smi).HasSubstructMatch(patt))

        table = [[proto_hits, len(proto_smiles) - proto_hits],
                 [bg_hits, len(bg_smiles) - bg_hits]]
        odds_ratio, p_val = fisher_exact(table)

        if p_val < 0.05 and odds_ratio > 1.2:
            print(f"  {name}: OR = {odds_ratio:.2f}, p = {p_val:.2e}, "
                  f"prototype {proto_hits}/{len(proto_smiles)}, background {bg_hits}/{len(bg_smiles)}")

print("\n" + "=" * 70)
print("Annotation complete.")