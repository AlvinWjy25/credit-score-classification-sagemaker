from data_ingestion import UnifiedCreditPreprocessor, CreditDataCleaning
from train import CreditFinalModel
from evaluation import EvaluateModel
from config import RAW_DATA_PATH, TARGET_COLUMN, PROCESSED_DATA_PATH, ARTIFACT_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from config import rfr_param_dist, xgb_param_dist, lgbm_param_dist

import traceback
from IPython.display import display
import mlflow

import warnings
warnings.filterwarnings("ignore") #debug


if __name__ == "__main__":
    #=== Data_ingestion ===
    try:

        preprocessor = UnifiedCreditPreprocessor(RAW_DATA_PATH, TARGET_COLUMN)
        preprocessor.fit(PROCESSED_DATA_PATH, TARGET_COLUMN, ver=0)  
        
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(test_size=0.2)
        print("\n[data_ingestion.py]: Pipeline berhasil!")
        print(f"[data_ingestion.py]: Shape X_train: {X_train.shape}")
        print(f"[data_ingestion.py]: Shape X_test: {X_test.shape}")
        
    except Exception as e:
        print(traceback.print_exc())
        raise ValueError(f"\n[data_ingestion.py] Terjadi error pada pipeline: {e}")
        
    print(f"="*100, "\n")


    #=== Train.py ===
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
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

    #=== Evaluate.py ===
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    try:
        evaluator = EvaluateModel(random_state=42)
        evaluator.load_saved_model()
        report_df = evaluator.evaluate_and_log(X_train, X_test, y_train, y_test)
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"\n[Pipeline.py] Terjadi error pada proses evaluation.py: {e}")
        
        
    
