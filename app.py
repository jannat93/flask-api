from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import requests

app = Flask(__name__)
CORS(app)

# ==== STEP 1: Download the model if not present ====
MODEL_ID = '1fAQVs4w8LUQnpFmkcgbXFU-Qpzv-CXHA'  # 👈 Replace with your Google Drive file ID
MODEL_URL = f'https://drive.google.com/uc?export=download&id={MODEL_ID}'
MODEL_PATH = 'har_voting_model.pkl'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print("Model downloaded successfully.")

download_model()

# ==== STEP 2: Load your saved model ====
model = joblib.load(MODEL_PATH)

# ==== STEP 3: Define prediction route ====
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

# ==== STEP 4: Run the app ====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
