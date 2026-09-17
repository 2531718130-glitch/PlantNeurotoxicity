"""
config.py
Centralized configuration for the PlantNeurotoxicity project.
All file paths are defined here as relative paths so that the project
can be run on any machine without modification.
"""

import os

# ==================== Base directory ====================
# The root directory of the project (i.e., the folder containing this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== Data paths ====================
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "step3_final_v4.csv")

# ==================== Output paths ====================
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation_v4")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# ==================== Model save paths ====================
# Use MODELS_DIR consistently. Keep MODEL_DIR as an alias for backward compatibility.
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_DIR = MODELS_DIR  # alias
os.makedirs(MODELS_DIR, exist_ok=True)

# Classical ML models
MODEL_RF_XGB = os.path.join(MODELS_DIR, "model_rf_xgb.pkl")
MODEL_SVM = os.path.join(MODELS_DIR, "model_svm.pkl")
MODEL_KNN = os.path.join(MODELS_DIR, "model_knn.pkl")
MODEL_NB = os.path.join(MODELS_DIR, "model_nb.pkl")
MODEL_STACKING = os.path.join(MODELS_DIR, "model_stacking.pkl")
MODEL_VOTING = os.path.join(MODELS_DIR, "model_voting.pkl")

# GNN models
MODEL_GNN_5FOLD = os.path.join(MODELS_DIR, "best_gnn_5fold.pt")
MODEL_DTRL_GNN_5FOLD = os.path.join(MODELS_DIR, "best_dtrl_gnn_5fold.pt")
# The MMP analysis uses the temperature-scaled GNN checkpoint
MODEL_GNN_5FOLD_V3 = os.path.join(MODELS_DIR, "best_gnn_5fold_v3_1.pt")

# ==================== Global random seed ====================
RANDOM_STATE = 42
N_FOLDS = 5

# ==================== Feature configuration ====================
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
N_DESCRIPTORS = 10
# Total feature dimension = 2048 (Morgan) + 10 (descriptors) = 2058
FEATURE_DIM = MORGAN_NBITS + N_DESCRIPTORS

# ==================== Train/Val/Test split ====================
TEST_SIZE = 0.15
VAL_SIZE = 0.176  # 0.176 * 0.85 ≈ 0.15 of the full dataset