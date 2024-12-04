from flask import Blueprint, request, jsonify
from model_utils import load_model, load_scaler, load_imputers
from preprocessing import preprocess_data
from config import FEATURE_ORDER
import logging

logging.basicConfig(level=logging.INFO)

# Blueprint for routes
main_blueprint = Blueprint('main', __name__)

# Load model and preprocessing objects
model = load_model()
scaler = load_scaler()
imputers = load_imputers()

@main_blueprint.route('/diagnosis', methods=['POST'])
def diagnose():
    try:
        logging.info("Received request: %s", request.json)
        
        # Parse input data
        data = request.json
        if 'patient_data' not in data:
            logging.error("Missing 'patient_data' key in request")
            return jsonify({"error": "Missing 'patient_data' key in request"}), 400

        patient_data = data['patient_data']
        logging.info("Patient data: %s", patient_data)

        # Preprocess the input data
        preprocessed_data = preprocess_data(patient_data, FEATURE_ORDER, scaler, imputers)
        logging.info("Preprocessed data shape: %s", preprocessed_data.shape)

        # Make predictions
        prediction = model.predict(preprocessed_data)
        logging.info("Prediction: %s", prediction)

        # Get probabilities if available
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(preprocessed_data)
            confidence = (probabilities * 100).round(2).tolist()
            logging.info("Prediction probabilities: %s", confidence)

        # Interpret prediction
        result = "ckd - YES" if prediction[0] == 1 else "ckd - NO"
        logging.info("Diagnosis result: %s", result)

        # Return the result
        return jsonify({"diagnosis": result, "confidence": confidence})

    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        return jsonify({"error": str(e)}), 500
