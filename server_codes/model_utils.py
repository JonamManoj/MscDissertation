import os
import joblib
from config import MODEL_DIR, PREPROCESSING_DIR

def load_model():
    """Load the trained model."""
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    return joblib.load(model_path)

def load_scaler():
    """Load the scaler for numerical features."""
    scaler_path = os.path.join(PREPROCESSING_DIR, "scaler.pkl")
    return joblib.load(scaler_path)

def load_imputers():
    """Load imputers for handling missing values."""
    imputers_path = os.path.join(PREPROCESSING_DIR, "imputers.pkl")
    return joblib.load(imputers_path)
