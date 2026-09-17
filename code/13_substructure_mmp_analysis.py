"""
mmp_analysis.py

Matched molecular pair (MMP) analysis with 5-fold GNN ensemble predictions.

This script performs MMP pairing, calls an external GNN prediction function,
and saves the pair-level results. It does NOT define the GNN model itself.

Usage:
    from your_gnn_module import predict_ensemble  # or similar
    run_mmp_analysis(csv_path, predict_func, output_dir)
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from scipy import stats
from scipy.stats import fisher_exact
import config as cfg

# ==================== Paths ====================
BASE_DIR = getattr(cfg, "BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = getattr(cfg, "DATA_DIR", os.path.join(BASE_DIR, "data"))
RESULTS_DIR = getattr(cfg, "RESULTS_DIR", os.path.join(BASE_DIR, "results"))


def improved_mmp_experiment(df_path, predict_func, output_dir,
                            max_pairs=50, mw_tol=50, tanimoto_tol=0.3):
    """
    Run MMP analysis.

    Parameters
    ----------
    df_path : str
        Path to CSV with columns 'smiles' and 'label'.
    predict_func : callable
        Function that takes a SMILES string and returns a predicted probability.
        Example: def predict(smiles): return prob, _, _, _  # only first value used
        It must return a float as the first element.
    output_dir : str
        Directory to save the output CSV.
    max_pairs : int
        Maximum number of matched pairs to generate.
    mw_tol : float
        Maximum allowed molecular weight difference (Da).
    tanimoto_tol : float
        Minimum Tanimoto similarity for fallback matching.

    Returns
    -------
    pd.DataFrame
        DataFrame containing pair-level results.
    """
    print("=" * 70)
    print("【MMP analysis】")
    print("=" * 70)

    df = pd.read_csv(df_path)

    pyrido_smarts = 'n1cnc2[c,n]cccc2[c,n]1'
    pyrido_patt = Chem.MolFromSmarts(pyrido_smarts)

    exclusion_patterns = {
        'Quinoline': 'c1ccc2ncccc2c1',
        'Indole': 'c1ccc2[nH]ccc2c1',
        'Pyrazole': 'n1[nH]ccc1',
        'Sulfonamide': '[S](=O)(=O)[N,n]',
    }
    exclusion_patts = {k: Chem.MolFromSmarts(v) for k, v in exclusion_patterns.items()}

    def calc_props(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        scaffold = MS.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else None
        return {'mol': mol, 'mw': mw, 'logp': logp, 'tpsa': tpsa, 'fp': fp,
                'scaffold': scaffold_smiles, 'smiles': smiles}

    print("Precomputing molecular properties...")
    props_list = []
    for _, row in df.iterrows():
        p = calc_props(row['smiles'])
        if p:
            p['label'] = row['label']
            p['has_pyrido'] = p['mol'].HasSubstructMatch(pyrido_patt) if pyrido_patt else False
            p['has_exclusion'] = any(p['mol'].HasSubstructMatch(ep) for ep in exclusion_patts.values() if ep)
            props_list.append(p)

    props_df = pd.DataFrame(props_list)

    # Group A: contains pyridopyrimidine and toxic
    group_a = props_df[(props_df['has_pyrido'] == True) & (props_df['label'] == 1)].copy()
    # Group B candidates: no pyridopyrimidine, non-toxic, and no other high-risk fragments
    group_b = props_df[(props_df['has_pyrido'] == False) &
                       (props_df['label'] == 0) &
                       (props_df['has_exclusion'] == False)].copy()

    print(f"\nGroup A (pyridopyrimidine + toxic): {len(group_a)}")
    print(f"Group B candidates (no pyrido + non-toxic + no confounders): {len(group_b)}")

    results = []
    used_b_idx = set()

    for _, a_row in group_a.iterrows():
        if len(results) >= max_pairs:
            break

        # 1. Try to match by Murcko scaffold
        scaffold_matches = group_b[group_b['scaffold'] == a_row['scaffold']]
        if len(scaffold_matches) == 0:
            # Fallback: Tanimoto similarity
            a_fp = a_row['fp']
            group_b['tanimoto'] = group_b['fp'].apply(
                lambda b_fp: AllChem.DataStructs.TanimotoSimilarity(a_fp, b_fp) if b_fp else 0
            )
            candidates = group_b[group_b['tanimoto'] >= tanimoto_tol]
        else:
            candidates = scaffold_matches.copy()
            candidates['tanimoto'] = 1.0

        # Exclude already used
        candidates = candidates[~candidates.index.isin(used_b_idx)]
        if len(candidates) == 0:
            continue

        # 2. Multi-dimensional distance score
        candidates['mw_diff'] = abs(candidates['mw'] - a_row['mw'])
        candidates['logp_diff'] = abs(candidates['logp'] - a_row['logp'])
        candidates['score'] = (candidates['mw_diff'] / 50.0 +
                               candidates['logp_diff'] / 2.0 +
                               (1 - candidates['tanimoto']))

        candidates = candidates[candidates['mw_diff'] <= mw_tol]
        if len(candidates) == 0:
            continue

        best_b = candidates.nsmallest(1, 'score').iloc[0]
        b_idx = best_b.name
        used_b_idx.add(b_idx)

        # 3. Predict probabilities using external function
        prob_a = predict_func(a_row['smiles'])
        prob_b = predict_func(best_b['smiles'])

        if prob_a is None or prob_b is None:
            continue

        delta = prob_a - prob_b

        results.append({
            'pair_id': len(results) + 1,
            'A_smiles': a_row['smiles'],
            'B_smiles': best_b['smiles'],
            'A_MW': round(a_row['mw'], 2),
            'B_MW': round(best_b['mw'], 2),
            'A_LogP': round(a_row['logp'], 2),
            'B_LogP': round(best_b['logp'], 2),
            'Tanimoto': round(best_b['tanimoto'], 3),
            'same_scaffold': a_row['scaffold'] == best_b['scaffold'],
            'A_prob': round(prob_a, 4),
            'B_prob': round(prob_b, 4),
            'delta': round(delta, 4)
        })

    df_result = pd.DataFrame(results)

    if len(df_result) > 0:
        deltas = df_result['delta'].values
        if len(deltas) >= 3:
            _, p_value = stats.wilcoxon(deltas, alternative='greater')
            cohens_d = np.mean(deltas) / np.std(deltas, ddof=1) if np.std(deltas) > 0 else 0

            print(f"\n{'='*70}")
            print(f"【MMP statistics】pairs: {len(df_result)}")
            print(f"  Mean Δ ± SD: {np.mean(deltas):.4f} ± {np.std(deltas):.4f}")
            print(f"  Median Δ: {np.median(deltas):.4f}")
            print(f"  Positive differences: {(deltas > 0).sum()}/{len(deltas)} ({(deltas>0).mean()*100:.1f}%)")
            print(f"  Wilcoxon p-value: {p_value:.2e}")
            print(f"  Cohen's d: {cohens_d:.3f}")
            print(f"{'='*70}")

            if len(deltas) > 5:
                min_idx = np.argmin(deltas)
                deltas_excl = np.delete(deltas, min_idx)
                _, p_excl = stats.wilcoxon(deltas_excl, alternative='greater')
                print(f"\n【Sensitivity analysis】excluding smallest delta:")
                print(f"  Mean Δ: {np.mean(deltas_excl):.4f} | p: {p_excl:.2e}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pair_experiment_v4.csv")
    df_result.to_csv(out_path, index=False)
    print(f"\nSaved MMP results to: {out_path}")

    return df_result


if __name__ == "__main__":
    # Example: import your existing prediction function
    # from your_gnn_module import predict_ensemble  # <- replace with your actual module
    # Then define a simple wrapper that returns only the probability
    # def my_predict(smiles):
    #     prob, _, _, _ = predict_ensemble(smiles, fold_states, fold_temps, scaler, dims)
    #     return prob
    #
    # csv_path = os.path.join(DATA_DIR, "step3_final_v4.csv")
    # output_dir = os.path.join(RESULTS_DIR, "validation_v4")
    # improved_mmp_experiment(csv_path, my_predict, output_dir)
    pass