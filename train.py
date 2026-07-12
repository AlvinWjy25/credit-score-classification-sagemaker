import pandas as pd
import numpy as np
import re
import os
import joblib
import mlflow

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

import traceback
from IPython.display import display
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_validate

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config import ARTIFACT_DIR, PROCESSED_DATA_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from config import rfr_param_dist, xgb_param_dist, lgbm_param_dist
from evaluation import EvaluateModel

class CreditFinalModel(EvaluateModel):
    def __init__(self, model_type, random_state=42):
        super().__init__(random_state=random_state) #inherit evaluation.py (evaluate_and_log)
        self.random_state = random_state
        self.model_type = model_type
        self.best_model = None
        self.cv_results = None

        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #self.scoring_metrics sudah di inherit oleh evaluation
       
        if model_type == 'rfc':
            self.base_model = RandomForestClassifier(n_estimators = 200, class_weight = "balanced_subsample",
                                                     random_state=self.random_state, n_jobs = -1)
                
        elif model_type == 'xgb':
            self.base_model = XGBClassifier(n_estimators=200, random_state = 42, n_jobs = -1)
        
        elif model_type == 'lgbm':
            self.base_model = LGBMClassifier(n_estimators=200, random_state = 42, n_jobs = 1,verbose= -1)

        else:
            raise KeyError(f"Make sure model_type: '{self.model_type}' exist.")
                

    def tune_hyperparameters(self, X_train, y_train, param_dist, cv = 5):
        print(f"\n\n[train.py]: Starting Hyperparameter Tuning using GridSearchCV...")

        self.X_train = X_train
        self.y_train = y_train

        if self.model_type == "rfc":
            print(f"Training Random Forest Classifier (4-7 minutes)")

        elif self.model_type == "xgb":
            print(f"Training XGBClassifier (4-7 minutes)")
        
        elif self.model_type == "lgbm":
            print(f"Training LGBMClassifier (4-7 minutes)")

        with mlflow.start_run(run_name=f"Tuning_{self.model_type.upper()}"):
            search = GridSearchCV(
                estimator=self.base_model,
                param_grid=param_dist,
                cv = cv,
                scoring = "f1_macro",
                n_jobs=4,
                verbose=1
            )

            search.fit(self.X_train, self.y_train)
            self.best_model = search.best_estimator_
            self.cv_results = search.cv_results_

            print(f"[train.py]: Best Parameters: {search.best_params_}")
            mlflow.log_params(search.best_params_)

            if hasattr(self, 'scaler'):
                mlflow.log_artifact(PROCESSED_DATA_PATH / "artifacts" / "best_model.joblib")
                mlflow.log_artifact(PROCESSED_DATA_PATH / "artifacts" / "scaler.joblib")
            
            return self.best_model
    
    def save_model(self, rfc_score, xgb_score, lgbm_score, filepath):
        """Menyimpan model ke lokal HANYA jika model ini adalah yang terbaik di antara ketiganya"""
        if self.best_model is None:
            print(f"[{self.model_type.upper()}]: No trained model found to evaluate for saving.")
            return

        mean_rfc = np.mean(rfc_score) if rfc_score is not None else -1
        mean_xgb = np.mean(xgb_score) if xgb_score is not None else -1
        mean_lgbm = np.mean(lgbm_score) if lgbm_score is not None else -1

        scores_dict = {
            'rfc': mean_rfc,
            'xgb': mean_xgb,
            'lgbm': mean_lgbm
        }

        self.best_model_type = max(scores_dict, key=scores_dict.get)
        best_score = scores_dict[self.best_model_type]

        path_obj = Path(filepath)
        if path_obj.suffix != '.joblib':
            path_obj = path_obj / "best_model.joblib"
        
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if self.model_type == self.best_model_type:
            joblib.dump(self.best_model, str(path_obj), compress=('lz4', 8))
            print(f"SUCCESS: [{self.model_type.upper()}, F1-Macro: {best_score:.4f}] is the overall best model and successfully saved to '{path_obj}'")
        else:
            print(f"SKIP: [{self.model_type.upper()}] is not saved because {self.best_model_type.upper()} performed better.")

        return self.best_model_type
        

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train = pd.read_csv(PROCESSED_DATA_PATH / "ingested"/ "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_PATH / "ingested"/ "X_test.csv")
    y_train = np.ravel(pd.read_csv(PROCESSED_DATA_PATH / "ingested"/ "y_train.csv"))
    y_test = np.ravel(pd.read_csv(PROCESSED_DATA_PATH / "ingested"/ "y_test.csv"))

    experiments = {
        'rfc': rfr_param_dist,
        'xgb': xgb_param_dist,
        'lgbm': lgbm_param_dist
    }

    scores = {'rfc': None, 'xgb': None, 'lgbm': None}
    trained_instances = {}

    for m_type, p_dist in experiments.items():
        model_runner = CreditFinalModel(model_type=m_type, random_state=42)
        model_runner.tune_hyperparameters(X_train, y_train, p_dist, cv=5)
        report_df = model_runner.evaluate_and_log(X_train, X_test, y_train, y_test, best_model=model_runner.best_model)
        
        cv_f1_macro_mean = report_df.loc[report_df['Metric'] == 'F1 Macro', 'CV Mean'].values[0]
        scores[m_type] = cv_f1_macro_mean
        trained_instances[m_type] = model_runner

    print(f"="*100, "\n")

    print("\n" + "#"*30 + " EXPORTING MODEL " + "#"*30)
    for m_type, instance in trained_instances.items():
        instance.save_model(
            rfc_score=scores['rfc'], 
            xgb_score=scores['xgb'], 
            lgbm_score=scores['lgbm'], 
            filepath=ARTIFACT_DIR
        )


    

