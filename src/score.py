import os
import json
import joblib
import numpy as np


def init():
    global model
    model_dir = os.environ["AZUREML_MODEL_DIR"]

    model_path = None
    for root, dirs, files in os.walk(model_dir):
        if "model.pkl" in files:
            model_path = os.path.join(root, "model.pkl")
            break

    if model_path is None:
        all_files = []
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                all_files.append(os.path.join(root, f))
        raise FileNotFoundError(
            f"model.pkl not found in {model_dir}. Files: {all_files}"
        )

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully")


def run(raw_data):
    data = json.loads(raw_data)
    features = np.array(data["data"])
    predictions = model.predict(features)
    return predictions.tolist()