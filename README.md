# Interpretable Machine Learning for Neurotoxicity Risk Prioritization of Plant-Derived Environmental Contaminants

Code and data for the manuscript: **The Pyrido[2,3-d]pyrimidine Toxicophore**.

## Overview

This repository contains the complete computational workflow for an interpretable machine-learning framework that predicts neurotoxicity of plant-derived environmental contaminants. The framework combines an optimized stacking ensemble with an attention-based graph neural network, extended by disentangled toxicity representation learning (DTRL), toxicity prototype discovery (ToxProto), and conformal prediction.

The central finding is the convergent identification of the pyrido[2,3-d]pyrimidine fused ring as a dominant structural determinant of neurotoxicity.

## Requirements

Python 3.8 or later is required. Install dependencies with:

```bash
pip install -r requirements.txt
Main dependencies:

numpy, pandas, scipy

scikit-learn

matplotlib

rdkit

torch, torch-geometric

shap

Data
Input data are stored in data/:

data/step3_final_v4.csv — curated dataset with columns smiles and label (0 = non-toxic, 1 = toxic).

Precomputed results used for figures are stored in results/validation_v4/.

## Execution order

Scripts in `code/` are numbered in the order they should be run:

1. `01_descriptor_distribution.py` — Descriptor distribution (Fig. 2)
2. `02_rf_xgb_training.py` — Random Forest and XGBoost training
3. `03_svm_5fold.py` — SVM 5-fold cross-validation
4. `04_knn_5fold.py` — KNN 5-fold cross-validation
5. `05_nb_5fold.py` — Naive Bayes 5-fold cross-validation
6. `06_stacking.py` — Stacking ensemble
7. `07_voting.py` — Voting ensemble
8. `08_model_comparison_plots.py` — Model comparison (Figs. 3–4)
9. `09_confusion_heatmap.py` — Confusion matrices and disagreement heatmap (Figs. 5–6)
10. `10_gnn_5fold.py` — Base GNN 5-fold training
11. `11_dtrl_gnn_5fold.py` — DTRL-GNN 5-fold training
12. `12_dtrl_representation_analysis.py` — DTRL subspace analysis
13. `13_substructure_mmp_analysis.py` — Substructure enrichment + MMP + attention correlation
14. `14_mmp_visualization.py` — MMP visualization (Fig. 12)
15. `15_gnn_attention_visualization.py` — GNN attention mapping (Fig. 9)
16. `16_shap_analysis.py` — SHAP value computation
17. `17_shap_barplot.py` — SHAP plots (Figs. 7–8)
18. `18_toxproto.py` — ToxProto prototype discovery
19. `19_toxproto_annotation.py` — ToxProto substructure annotation
20. `20_toxproto_visualization.py` — ToxProto visualization (Fig. 13)
21. `21_conformal_prediction.py` — Conformal prediction
22. `22_conformal_visualization.py` — Conformal prediction visualization (Fig. 14)
To reproduce all figures and tables from scratch, run the scripts in numerical order:

bash

cd code
python 01_descriptor_distribution.py
python 02_rf_xgb_training.py
...
python 22_conformal_visualization.py

Repository structure

text

PlantNeurotoxicity/
├── code/                    # All analysis scripts
├── data/                    # Input datasets
├── models/                  # Trained model checkpoints
├── results/                 # Output figures and tables
│   └── validation_v4/       # Precomputed MMP and substructure results
├── supplementary/           # Supporting information
├── config.py                # Centralized path configuration
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md                # This file
Citation
If you use this code or data in your work, please cite:

Ma H, Ji Z, Han W. Interpretable Machine Learning for Neurotoxicity Risk Prioritization of Plant-Derived Environmental Contaminants: The Pyrido[2,3-d]pyrimidine Toxicophore. Environ. Sci. Technol. (submitted).

The archived version of this repository is available on Zenodo:

> https://doi.org/10.5281/zenodo.22811605

License
This project is licensed under the MIT License. See LICENSE for details.

Contact
For questions or issues, please open a GitHub issue or contact:

Weiwei Han — weiweihan@jlu.edu.cn