import os
import logging
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
import tensorflow as tf

# Configure logging to display progress and debugging information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CKDDiagnosisServer:
    def __init__(self):
        """
        Initializes the CKDDiagnosisServer class with Flask application and model paths.
        """
        logging.info("Initializing CKDDiagnosisServer...")
        self.app = Flask(__name__)
        self.setup_routes()

        # Base directory and model paths
        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.model_files_directory = os.path.abspath(os.path.join(base_dir, "..", "model_training_output_files"))
        self.preprocessing_dir = os.path.join(self.model_files_directory, "preprocessing")

        # Feature order
        self.feature_order = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc',
            'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]

        # Load all models and preprocessing artifacts
        self.models = {}
        self.load_models()

    def load_models(self):
        """
        Load all trained models and preprocessing artifacts.
        """
        logging.info("Loading models and preprocessing artifacts...")
        # Load Sklearn models
        for model_name in [
            'Logistic Regression', 'SVM', 'Random Forest', 'XGBoost'
        ]:
            model_file_path = os.path.join(self.model_files_directory, f"{model_name}_model.pkl")
            if os.path.exists(model_file_path):
                self.models[model_name] = joblib.load(model_file_path)
                logging.info(f"Loaded Sklearn model: {model_name}")

        # Load Neural Network models
        for model_name in ['ANN', 'CNN', 'RNN', 'LSTM']:
            model_file_path = os.path.join(self.model_files_directory, f"{model_name}_model.keras")
            if os.path.exists(model_file_path):
                self.models[model_name] = tf.keras.models.load_model(model_file_path)
                logging.info(f"Loaded Neural Network model: {model_name}")

        # Load preprocessing artifacts
        scaler_path = os.path.join(self.preprocessing_dir, "scaler.pkl")
        imputers_path = os.path.join(self.preprocessing_dir, "imputers.pkl")
        self.scaler = joblib.load(scaler_path)
        self.imputers = joblib.load(imputers_path)
        logging.info("Loaded scaler and imputers.")

    def preprocess_data(self, input_data):
        """
        Preprocess input data for the model.
        Parameters:
            input_data (dict): Dictionary containing patient features.
        Returns:
            np.array: Scaled feature array ready for prediction.
        """
        logging.info("Preprocessing input data...")
        input_df = pd.DataFrame([input_data], columns=self.feature_order)

        numerical_cols = [
            'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot',
            'hemo', 'pcv', 'wbcc', 'rbcc'
        ]
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

        # Impute missing values for numerical features
        for col in numerical_cols:
            if col in self.imputers:
                input_df[[col]] = self.imputers[col].transform(input_df[[col]])

        # Map categorical data to numeric codes
        category_mapping = {
            None: 0, "": 0,
            "normal": 1, "present": 1, "yes": 1, "good": 1,
            "abnormal": 2, "notpresent": 2, "no": 2, "poor": 2
        }
        input_df[categorical_cols] = input_df[categorical_cols].applymap(lambda val: category_mapping.get(val, 0))

        # Scale the numerical features
        scaled_data = self.scaler.transform(input_df)

        return scaled_data

    def setup_routes(self):
        """
        Set up routes for the Flask application.
        """
        @self.app.route('/diagnosis', methods=['POST'])
        def diagnose():
            try:
                logging.info("Received request: %s", request.json)
                data = request.json
                
                if 'patient_data' not in data:
                    logging.error("Missing 'patient_data' key in request")
                    return jsonify({"error": "Missing 'patient_data' key in request"}), 400

                if 'model_name' not in data:
                    logging.error("Missing 'model_name' key in request")
                    return jsonify({"error": "Missing 'model_name' key in request"}), 400

                patient_data = data['patient_data']
                model_name = data['model_name']
                logging.info("Patient data: %s", patient_data)
                logging.info("Requested model: %s", model_name)

                # Preprocess the input data
                preprocessed_data = self.preprocess_data(patient_data)
                logging.info("Preprocessed data shape: %s", preprocessed_data.shape)

                # Check if the specified model exists
                if model_name not in self.models:
                    logging.error("Model '%s' not found", model_name)
                    return jsonify({"error": f"Model '{model_name}' not found"}), 400

                # Make predictions with the specified model
                model = self.models[model_name]
                if model_name in ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost']:
                    prediction = model.predict(preprocessed_data)[0]
                    result = "ckd - YES" if prediction == 1 else "ckd - NO"
                    confidence = None
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(preprocessed_data)[0]
                        confidence = [round(float(prob), 2) for prob in (probabilities * 100)]

                elif model_name in ['ANN', 'CNN', 'RNN', 'LSTM']:
                    prediction = np.argmax(model.predict(preprocessed_data), axis=1)[0]
                    result = "ckd - YES" if prediction == 1 else "ckd - NO"
                    probabilities = model.predict(preprocessed_data)[0]
                    confidence = [round(float(prob), 2) for prob in (probabilities * 100)]

                logging.info("Prediction: %s", result)
                logging.info("Confidence: %s", confidence)

                # Return the result
                return jsonify({"diagnosis": result, "confidence": confidence})

            except Exception as e:
                logging.error("Error occurred: %s", str(e))
                return jsonify({"error": str(e)}), 500

    def run_server(self):
        """
        Run the Flask server to provide prediction services.
        """
        logging.info("Starting Flask server...")
        self.app.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    # Initialize and run the server
    server = CKDDiagnosisServer()
    server.run_server()
