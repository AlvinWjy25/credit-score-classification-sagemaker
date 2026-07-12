from pathlib import Path
import os

#Current Working Directory
ROOT_DIR = Path(__file__).resolve().parent.parent

#Common use
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "data_C.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" 

#Make Dir Auto
ROOT_MAKE = ROOT_DIR/ "data"
RAW_DIR = ROOT_DIR / "data" / "raw"
ARTIFACT_DIR = ROOT_DIR / "data" / "artifacts"
INGESTED_DIR = PROCESSED_DATA_PATH / "ingested"
REPORT_DIR = PROCESSED_DATA_PATH / "report"

MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"

ROOT_MAKE.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
INGESTED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

#Target Model
TARGET_COLUMN = "Credit_Score"

#Defining cols
NUM_COLS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", 
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate"
]

CAT_COLS = [
    "Occupation", "Payment_of_Min_Amount", "Payment_Behaviour"
]

#Absolute Value

RANDOM_STATE = 42
CV_FOLDS = 5

rfr_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 4, 6],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', 'balanced_subsample']
}

xgb_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.1],
    'tree_method': ['hist']
}

lgbm_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1]
}


MLFLOW_EXPERIMENT_NAME = "Credit_Score_Classification"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

