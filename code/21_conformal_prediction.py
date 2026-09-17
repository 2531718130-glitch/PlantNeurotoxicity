"""
21_conformal_prediction.py
Inductive Conformal Prediction on Stacking model OOF probabilities.

Quantifies per-molecule prediction reliability at the 90% confidence level,
enabling tiered risk stratification.

Input:  results/stacking_prob_stacking_oof.npy
        results/stacking_label_oof.npy
Output: results/conformal_results.npz
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def conformal_prediction(calib_probs, calib_labels, test_probs, alpha=0.1):
    """
    Inductive Conformal Prediction.

    Parameters
    ----------
    calib_probs : array, predicted probabilities on the calibration set
    calib_labels : array, true labels on the calibration set
    test_probs : array, predicted probabilities on the test set
    alpha : float, significance level (1 - alpha = confidence level)

    Returns
    -------
    prediction_sets : list of sets, one per test sample
    p_values : array, conformal p-values
    q_hat : float, calibrated quantile threshold
    """
    n_calib = len(calib_probs)

    # Nonconformity scores: |y_i - p_i|
    cal_scores = np.abs(calib_labels - calib_probs)

    # Quantile threshold
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_hat = np.quantile(cal_scores, q_level, method='higher')

    # Compute p-values for test samples
    p_values = np.zeros(len(test_probs))
    for i, p in enumerate(test_probs):
        p_values[i] = (1 + np.sum(cal_scores >= np.abs(1 - p))) / (n_calib + 1)

    # Build prediction sets
    prediction_sets = []
    for p_val in p_values:
        if p_val > alpha:
            prediction_sets.append({1})
        else:
            prediction_sets.append({0})

    return prediction_sets, p_values, q_hat


if __name__ == "__main__":
    print("=" * 70)
    print("Inductive Conformal Prediction on Stacking OOF probabilities")
    print("=" * 70)

    prob_stack = np.load(os.path.join(config.RESULTS_DIR, 'stacking_prob_stacking_oof.npy'))
    labels = np.load(os.path.join(config.RESULTS_DIR, 'stacking_label_oof.npy'))

    print(f"Samples: {len(prob_stack)}")
    print(f"Toxicity prevalence: {labels.mean():.3f}")

    # Split into calibration (50%) and test (50%) sets
    np.random.seed(config.RANDOM_STATE)
    n = len(prob_stack)
    idx = np.random.permutation(n)
    n_calib = n // 2
    calib_idx = idx[:n_calib]
    test_idx = idx[n_calib:]

    calib_probs = prob_stack[calib_idx]
    calib_labels = labels[calib_idx]
    test_probs = prob_stack[test_idx]
    test_labels = labels[test_idx]

    print(f"Calibration set: {len(calib_probs)}")
    print(f"Test set: {len(test_probs)}")

    alpha = 0.1
    pred_sets, p_values, q_hat = conformal_prediction(
        calib_probs, calib_labels, test_probs, alpha=alpha
    )

    # Coverage
    coverage = np.mean([test_labels[i] in pred_sets[i] for i in range(len(test_labels))])
    set_sizes = np.array([len(s) for s in pred_sets])

    print(f"\n{'=' * 55}")
    print(f"Conformal Prediction Results (alpha = {alpha}, confidence = {1 - alpha:.0%})")
    print(f"{'=' * 55}")
    print(f"Empirical coverage: {coverage:.3f} (expected ≈ {1 - alpha:.2f})")
    print(f"Singleton prediction set rate: {np.mean(set_sizes == 1):.1%}")
    print(f"Double prediction set rate:    {np.mean(set_sizes == 2):.1%}")

    # High-uncertainty compounds
    uncertainty = np.abs(p_values - alpha)
    high_uncertainty_idx = np.argsort(uncertainty)[:20]
    print(f"\nTop 20 most uncertain compounds (p-value closest to {alpha}):")
    for i in high_uncertainty_idx:
        print(f"  Test index {test_idx[i]}: prob = {test_probs[i]:.3f}, "
              f"p = {p_values[i]:.4f}, true label = {test_labels[i]:.0f}")

    np.savez(os.path.join(config.RESULTS_DIR, 'conformal_results.npz'),
             calib_idx=calib_idx,
             test_idx=test_idx,
             p_values=p_values,
             q_hat=q_hat,
             coverage=coverage,
             test_labels=test_labels,
             test_probs=test_probs)

    print(f"\nSaved: {os.path.join(config.RESULTS_DIR, 'conformal_results.npz')}")