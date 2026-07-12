import pandas as pd
import numpy as np
import joblib
import mlflow
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from config import ARTIFACT_DIR, PROCESSED_DATA_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

warnings.filterwarnings("ignore")

class EvaluateModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.scoring_metrics = {
            'accuracy': 'accuracy',
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'f1_micro': make_scorer(f1_score, average='micro')
        }
        
    def load_saved_model(self):
        """Method khusus untuk standalone execution: memuat model dari storage local"""
        try:
            print(f"[Evaluation.py]: Importing existing best model from {ARTIFACT_DIR}...")
            self.best_model = joblib.load(ARTIFACT_DIR / "best_model.joblib")
        except Exception as e:
            raise FileNotFoundError(f"[!] Model not found in {ARTIFACT_DIR}, please run train.py to export model first! Error: {e}")

    def evaluate_and_log(self, X_train, X_test, y_train, y_test, best_model=None):
        """
        Fungsi fleksibel:
        - Jika dijalankan dari train.py: menerima objek best_model langsung (in-memory).
        - Jika dijalankan mandiri: menggunakan self.best_model yang sudah dimuat dari local storage.
        """
        if best_model is not None:
            self.best_model = best_model
            
        if self.best_model is None:
            raise ValueError("No model available for evaluation. Provide a model or load one from storage.")
            
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Mendapatkan nama arsitektur model secara dinamis untuk label MLflow
        model_name = type(self.best_model).__name__

        with mlflow.start_run(run_name=f"BestModel_Evaluation_{model_name}", nested=True):
            print(f"[Evaluation.py]: Running Stratified 5-Fold Cross Validation for {model_name}...")
            cv_results = cross_validate(
                self.best_model, self.X_train, self.y_train, 
                cv=self.skf, scoring=self.scoring_metrics, n_jobs=-1
            )

            print("[Evaluation.py]: Predicting test set...")
            y_pred = self.best_model.predict(self.X_test)
            
            test_metrics = {
                "Accuracy": accuracy_score(self.y_test, y_pred),
                "Precision (Macro)": precision_score(self.y_test, y_pred, average='macro'),
                "Recall (Macro)": recall_score(self.y_test, y_pred, average='macro'),
                "F1 Macro": f1_score(self.y_test, y_pred, average='macro'),
                "F1 Micro": f1_score(self.y_test, y_pred, average='micro')
            }

            report_data = {
                "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 Macro", "F1 Micro"],
                "CV Mean": [
                    np.mean(cv_results.get('test_accuracy', [0])),
                    np.mean(cv_results.get('test_precision_macro', [0])),  
                    np.mean(cv_results.get('test_recall_macro', [0])),     
                    np.mean(cv_results.get('test_f1_macro', [0])),
                    np.mean(cv_results.get('test_f1_micro', [0]))
                ],
                "CV Std (+/-)": [
                    np.std(cv_results.get('test_accuracy', [0])),
                    np.std(cv_results.get('test_precision_macro', [0])),  
                    np.std(cv_results.get('test_recall_macro', [0])),     
                    np.std(cv_results.get('test_f1_macro', [0])),
                    np.std(cv_results.get('test_f1_micro', [0]))
                ]
            }
                
            self.report = pd.DataFrame(report_data)
            print("\n" + "="*20 + f" Final Test Set Evaluation ({model_name}) " + "="*20)
            print(classification_report(self.y_test, y_pred))
            
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))

            print("\n" + "="*20 + " Summary Report DataFrame " + "="*20)
            print(self.report.to_string(index=False))

            mlflow.log_metrics({
                "test_accuracy": test_metrics["Accuracy"],
                "cv_mean_accuracy": np.mean(cv_results['test_accuracy']),
                "test_f1_macro": test_metrics["F1 Macro"],
                "cv_mean_f1_macro": np.mean(cv_results['test_f1_macro']),
            })

            report_dir = PROCESSED_DATA_PATH / "report"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_file = report_dir / f"evaluation_report_{model_name}.csv"
            
            self.report.to_csv(report_file, index=False)
            mlflow.log_artifact(str(report_file))
            print(f"[Evaluation.py][MLFLOW]: Metrics and Report logged successfully!")
            print(f"="*66)
        
        return self.report


if __name__ == "__main__":
    print("\n" + "="*20 + " RUNNING EVALUATION INDEPENDENTLY " + "="*20)
    try:
        X_train_data = pd.read_csv(PROCESSED_DATA_PATH / "ingested" / "X_train.csv")
        X_test_data = pd.read_csv(PROCESSED_DATA_PATH / "ingested" / "X_test.csv")
        y_train_data = np.ravel(pd.read_csv(PROCESSED_DATA_PATH / "ingested" / "y_train.csv"))
        y_test_data = np.ravel(pd.read_csv(PROCESSED_DATA_PATH / "ingested" / "y_test.csv"))
        
        evaluator = EvaluateModel(random_state=42)
        evaluator.load_saved_model()
        evaluator.evaluate_and_log(X_train_data, X_test_data, y_train_data, y_test_data)
        
    except Exception as e:
        print(f"\n[EVALUATION MANDIRI GAGAL]: {e}")
