import logging
import os
import joblib
from keras.models import save_model
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

def save_models(models, nn_models, preprocessed_data, model_files_directory):
    logging.info("Saving all models and preprocessing artifacts...")
    os.makedirs(model_files_directory, exist_ok=True)

    # Save Sklearn models
    for name, model in models.items():
        model_file_path = os.path.join(model_files_directory, f"{name}_model.pkl")
        joblib.dump(model, model_file_path)
        logging.info(f"Saved Sklearn model: {name} at {model_file_path}")

    # Save Neural Network models
    for name, model in nn_models.items():
        model_file_path = os.path.join(model_files_directory, f"{name}_model.keras")
        model.save(model_file_path)
        logging.info(f"Saved Neural Network model: {name} at {model_file_path}")

    # Save preprocessing artifacts
    preprocessing_dir = os.path.join(model_files_directory, "preprocessing")
    os.makedirs(preprocessing_dir, exist_ok=True)

    # Save scaler
    scaler_path = os.path.join(preprocessing_dir, "scaler.pkl")
    joblib.dump(preprocessed_data["scaler"], scaler_path)
    logging.info(f"Saved scaler at {scaler_path}")

    # Save imputers
    imputers_path = os.path.join(preprocessing_dir, "imputers.pkl")
    joblib.dump(preprocessed_data["imputers"], imputers_path)
    logging.info(f"Saved imputers at {imputers_path}")

    # Save the best model
    best_model, best_model_name = get_best_model(models, nn_models, preprocessed_data)
    best_model_path = os.path.join(model_files_directory, f"best_model.pkl" if isinstance(best_model, GridSearchCV) else f"best_model_{best_model_name}.keras")
    if isinstance(best_model, GridSearchCV):
        joblib.dump(best_model.best_estimator_, best_model_path)
    else:
        best_model.save(best_model_path)
    logging.info(f"Saved best model: {best_model_name} at {best_model_path}")

    logging.info("All models and preprocessing artifacts saved successfully.")

def get_best_model(models, nn_models, preprocessed_data):
    logging.info("Determining the best model...")
    X_test = preprocessed_data["X_test_scaled"]
    y_test = preprocessed_data["y_test"]

    # Evaluate sklearn models
    best_model = None
    best_score = -float('inf')
    best_model_name = ""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)  # Accuracy metric
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name

    # Evaluate NN models
    for name, model in nn_models.items():
        y_pred = np.argmax(model.predict(X_test[..., None]), axis=1)
        score = accuracy_score(y_test, y_pred)
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name

    logging.info(f"The best model is {best_model_name} with a score of {best_score:.4f}")
    return best_model, best_model_name
