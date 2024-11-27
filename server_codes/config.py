import os

# Base code directory
base_dir = os.path.abspath(os.path.dirname(__file__))

# Paths for model and preprocessing artifacts
MODEL_DIR = os.path.abspath(os.path.join(base_dir, "..", "model_training_output_files"))
PREPROCESSING_DIR = os.path.join(MODEL_DIR, "preprocessing")

# Feature order
FEATURE_ORDER = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc',
    'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]
