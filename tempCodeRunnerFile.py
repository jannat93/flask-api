from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
# import logging # Optional: for logging

app = Flask(__name__)
CORS(app)

# Optional: Basic logging to see requests
# logging.basicConfig(level=logging.DEBUG)

# Load your saved model
# Ensure 'har_voting_model.pkl' is in the same directory or provide the correct path
try:
    model = joblib.load('har_voting_model.pkl')
    # app.logger.info("Model 'har_voting_model.pkl' loaded successfully.") # Optional logging
except FileNotFoundError:
    # app.logger.error("Model file 'har_voting_model.pkl' not found!") # Optional logging
    model = None # Handle case where model doesn't load
except Exception as e:
    # app.logger.error(f"Error loading model: {e}") # Optional logging
    model = None


@app.route('/predict', methods=['POST'])
def predict():
    # app.logger.debug("Received request at /predict") # Optional logging
    if model is None:
        # app.logger.error("Model not loaded, cannot predict.") # Optional logging
        return jsonify({'error': 'Model not loaded on server'}), 500

    try:
        data = request.get_json()
        if 'features' not in data or not isinstance(data['features'], list):
            # app.logger.warning("Invalid payload: 'features' key missing or not a list.") # Optional logging
            return jsonify({'error': "Invalid payload format: 'features' key missing or not a list"}), 400

        features_list = data['features']
        if not features_list or not isinstance(features_list[0], list):
             # app.logger.warning("Invalid payload: 'features' should be a list of lists.") # Optional logging
            return jsonify({'error': "Invalid payload format: 'features' should be a list of lists."}), 400

        features = np.array(features_list).reshape(1, -1) # Assuming a single sample per request for now
        # app.logger.debug(f"Features received for prediction: {features}") # Optional logging

        string_prediction = model.predict(features)[0] # model.predict returns an array, get the first element
        # app.logger.info(f"Prediction made: {string_prediction}") # Optional logging

        return jsonify({'prediction': string_prediction}) # Return the string label itself
    except Exception as e:
        # app.logger.error(f"Error during prediction: {e}") # Optional logging
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # app.logger.info("Starting Flask server...") # Optional logging
    app.run(host='0.0.0.0', port=5000, debug=True)