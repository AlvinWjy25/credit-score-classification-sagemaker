import os
import json
import joblib
import pandas as pd

# 1. Fungsi untuk me-load model (Wajib)
def model_fn(model_dir):
    """Me-load model dari directory bawaan SageMaker"""
    model_path = os.path.join(model_dir, "best_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return {"model": model, "scaler": scaler}

# 2. Fungsi untuk memproses input data (Wajib jika menggunakan JSON/Custom format)
def input_fn(request_body, request_content_type):
    """Mengubah payload input menjadi format yang dipahami model (misal: NumPy array atau Pandas DataFrame)"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return pd.DataFrame([data]) 
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# 3. Fungsi untuk melakukan prediksi (Opsional, tapi direkomendasikan)
def predict_fn(input_data, model_dict):
    model = model_dict['model']
    scaler = model_dict['scaler']

    scaled_data = scaler.transform(input_data)
    prediction_class = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)

    return {
        "class": prediction_class,
        "probabilities": prediction_proba
    }

# 4. Fungsi untuk memformat output (Opsional, tapi direkomendasikan)
def output_fn(prediction_result, content_type):
    if content_type == 'application/json':
        result_class = int(prediction_result["class"][0])
        result_proba = prediction_result["probabilities"][0].tolist()
        label_map = {0: "Poor", 1: "Standard", 2: "Good"}

        response = {
            "credit_score_prediction": label_map.get(result_class, "Unknown"),
            "class_index": result_class,
            "probabilities": result_proba 
        }

        return json.dumps(response), content_type
    
    raise ValueError(f"Unsupported content type: {content_type}")
