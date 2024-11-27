from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# Define the base directory (where your script is located)
base_dir = os.path.dirname(__file__)

# Define the model path
model_path = os.path.join(base_dir, "model_files")
preprocessing_dir = os.path.join(model_path, "preprocessing")

# Load the best model
best_model_path = os.path.join(model_path, "best_model.pkl")
best_model = joblib.load(best_model_path)

# Load the scaler for numerical data
scaler = joblib.load(os.path.join(preprocessing_dir, "scaler.pkl"))

# Load all imputers from a single file
imputers_path = os.path.join(preprocessing_dir, "imputers.pkl")
numerical_imputers = joblib.load(imputers_path)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the explicit feature order expected by the model
        feature_order = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc',
            'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]

        # Receive input data as JSON
        data = request.json
        raw_input = data['input']

        # Convert raw input to DataFrame with proper column names
        input_data_df = pd.DataFrame([raw_input], columns=feature_order)

        # Separate numerical and categorical features
        numerical_cols = [
            'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot',
            'hemo', 'pcv', 'wbcc', 'rbcc'
        ]
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

        # Impute missing values for numerical features
        for col in numerical_cols:
            if col in numerical_imputers:
                input_data_df[[col]] = numerical_imputers[col].transform(input_data_df[[col]])

        # Handle categorical data with mapping
        value_mapping = {
            None: 0, "": 0,
            "normal": 1, "present": 1, "yes": 1, "good": 1,
            "abnormal": 2, "notpresent": 2, "no": 2, "poor": 2
        }

        # Apply mapping to categorical columns
        input_data_df[categorical_cols] = input_data_df[categorical_cols].applymap(lambda x: value_mapping.get(x, 0))

        # Scale the data using the loaded scaler
        processed_input = scaler.transform(input_data_df)

        # Make prediction with the best model
        prediction = best_model.predict(processed_input)

        # Check if model has predict_proba (not all models do)
        if hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(processed_input)
            # Convert probabilities to percentages and round to 2 decimal places
            percentage_probabilities = (probabilities * 100).round(2).tolist()
        else:
            percentage_probabilities = None

        # Decode prediction (assuming binary classes where 1 = "ckd" and 0 = "notckd")
        decoded_prediction = "ckd - YES" if prediction[0] == 1 else "ckd - No"

        # Return the prediction and probabilities
        return jsonify({
            "prediction": decoded_prediction
        })

    except Exception as e:
        # Handle errors and return a meaningful response
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)



# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd
# import os

# # Path to saved preprocessing objects and model
# model_path = "./model_files"

# # Load all models
# models = {
#     "best_ckd_model": joblib.load(os.path.join(model_path, "best_ckd_model.pkl")),
#     "logistic_regression": joblib.load(os.path.join(model_path, "logistic_regression_model.pkl")),
#     "knn": joblib.load(os.path.join(model_path, "knn_model.pkl")),
#     "mlp": joblib.load(os.path.join(model_path, "mlp_(neural_network)_model.pkl")),
#     "random_forest": joblib.load(os.path.join(model_path, "random_forest_model.pkl")),
#     "svm": joblib.load(os.path.join(model_path, "svm_model.pkl")),
#     "xgboost": joblib.load(os.path.join(model_path, "xgboost_model.pkl")),
#     "worst_ckd_model": joblib.load(os.path.join(model_path, "worst_ckd_model.pkl")),
# }

# # Load the scaler for numerical data
# scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))

# # Load imputers for numerical columns
# numerical_imputers = {
#     "age": joblib.load(os.path.join(model_path, "imputer_age.pkl")),
#     "bp": joblib.load(os.path.join(model_path, "imputer_bp.pkl")),
#     "sg": joblib.load(os.path.join(model_path, "imputer_sg.pkl")),
#     "al": joblib.load(os.path.join(model_path, "imputer_al.pkl")),
#     "su": joblib.load(os.path.join(model_path, "imputer_su.pkl")),
#     "bgr": joblib.load(os.path.join(model_path, "imputer_bgr.pkl")),
#     "bu": joblib.load(os.path.join(model_path, "imputer_bu.pkl")),
#     "sc": joblib.load(os.path.join(model_path, "imputer_sc.pkl")),
#     "sod": joblib.load(os.path.join(model_path, "imputer_sod.pkl")),
#     "pot": joblib.load(os.path.join(model_path, "imputer_pot.pkl")),
#     "hemo": joblib.load(os.path.join(model_path, "imputer_hemo.pkl")),
#     "pcv": joblib.load(os.path.join(model_path, "imputer_pcv.pkl")),
#     "wbcc": joblib.load(os.path.join(model_path, "imputer_wbcc.pkl")),
#     "rbcc": joblib.load(os.path.join(model_path, "imputer_rbcc.pkl")),
# }

# # Initialize Flask app
# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Define the explicit feature order expected by the model
#         feature_order = [
#             'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc',
#             'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
#         ]

#         # Receive input data as JSON
#         data = request.json
#         raw_input = data['input']
#         selected_model = data.get('model', 'best_ckd_model').lower()  # Default to best model if not specified

#         # Ensure model name is valid
#         if selected_model not in models:
#             return jsonify({"error": f"Invalid model name '{selected_model}'. Available models: {list(models.keys())}"}), 400

#         # Convert raw input to DataFrame with proper column names
#         input_data_df = pd.DataFrame([raw_input], columns=feature_order)

#         # Separate numerical and categorical features
#         numerical_cols = [
#             'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot',
#             'hemo', 'pcv', 'wbcc', 'rbcc'
#         ]
#         categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

#         # Impute missing values for numerical features
#         for col in numerical_cols:
#             if col in numerical_imputers:
#                 input_data_df[[col]] = numerical_imputers[col].transform(input_data_df[[col]])

#         # Handle categorical data with mapping
#         value_mapping = {
#             None: 0, "": 0,
#             "normal": 1, "present": 1, "yes": 1, "good": 1,
#             "abnormal": 2, "notpresent": 2, "no": 2, "poor": 2
#         }

#         # Apply mapping to categorical columns
#         input_data_df[categorical_cols] = input_data_df[categorical_cols].applymap(lambda x: value_mapping.get(x, 0))

#         # Scale the data using the loaded scaler
#         processed_input = scaler.transform(input_data_df)

#         # Log the processed input for debugging purposes
#         # print("Processed Input:", processed_input)

#         # Make prediction with the selected model
#         model = models[selected_model]
#         prediction = model.predict(processed_input)

#         # Check if model has predict_proba (not all models do)
#         if hasattr(model, "predict_proba"):
#             probabilities = model.predict_proba(processed_input)
#             # Convert probabilities to percentages and round to 2 decimal places
#             percentage_probabilities = (probabilities * 100).round(2).tolist()
#         else:
#             percentage_probabilities = None

#         # Decode prediction (assuming binary classes where 1 = "ckd" and 0 = "notckd")
#         decoded_prediction = "ckd" if prediction[0] == 1 else "notckd"

#         # Return the prediction and probabilities
#         return jsonify({
#             "prediction": decoded_prediction,
#             "probabilities": percentage_probabilities
#         })

#     except Exception as e:
#         # Handle errors and return a meaningful response
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=9999)
