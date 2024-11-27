from flask import Blueprint, request, jsonify
from model_utils import load_model, load_scaler, load_imputers
from preprocessing import preprocess_data
from config import FEATURE_ORDER

# Blueprint for routes
main_blueprint = Blueprint('main', __name__)

# Load model and preprocessing objects
model = load_model()
scaler = load_scaler()
imputers = load_imputers()

@main_blueprint.route('/diagnosis', methods=['POST'])
def diagnose():
    try:
        # Parse input data from the request
        data = request.json
        patient_data = data['patient_data']

        # Preprocess the input data
        preprocessed_data = preprocess_data(patient_data, FEATURE_ORDER, scaler, imputers)

        # Make predictions
        prediction = model.predict(preprocessed_data)

        # Optionally get probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(preprocessed_data)
            confidence = (probabilities * 100).round(2).tolist()
        else:
            confidence = None

        # Interpret prediction
        result = "ckd - YES" if prediction[0] == 1 else "ckd - NO"

        # Return the result as JSON
        return jsonify({"diagnosis": result, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
