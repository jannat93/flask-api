from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Download the model if it doesn't exist
MODEL_PATH = 'har_voting_model.pkl'
MODEL_FILE_ID = '1fAQVs4w8LUQnpFmkcgbXFU-Qpzv-CXHA'  # Replace with your real file ID

if not os.path.exists(MODEL_PATH):
    import gdown
    gdown.download(f'https://drive.google.com/uc?id={MODEL_FILE_ID}', MODEL_PATH, quiet=False)

# Load the model
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
