import pandas as pd

def preprocess_data(input_data, feature_order, scaler, imputers):
    """Preprocess input data for the model."""
    # Convert raw input to DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_order)

    # Separate numerical and categorical features
    numerical_cols = [
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot',
        'hemo', 'pcv', 'wbcc', 'rbcc'
    ]
    categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    # Impute missing values for numerical features
    for col in numerical_cols:
        if col in imputers:
            input_df[[col]] = imputers[col].transform(input_df[[col]])

    # Map categorical data to numeric codes
    category_mapping = {
        None: 0, "": 0,
        "normal": 1, "present": 1, "yes": 1, "good": 1,
        "abnormal": 2, "notpresent": 2, "no": 2, "poor": 2
    }
    input_df[categorical_cols] = input_df[categorical_cols].applymap(lambda val: category_mapping.get(val, 0))

    # Scale the numerical features
    scaled_data = scaler.transform(input_df)

    return scaled_data
