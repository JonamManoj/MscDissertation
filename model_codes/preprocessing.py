import logging
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical

def preprocess_data(data_path):
    logging.info("Loading data from CSV...")
    data = pd.read_csv(data_path)

    logging.info("Handling missing values and encoding categorical columns...")
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if 'class' in categorical_cols:
        categorical_cols.remove('class')

    imputers = {}
    for col in numerical_cols:
        if data[col].isna().sum() > 0:
            strategy = 'median' if col in ['sg', 'al', 'su', 'bgr', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc'] else 'mean'
            imputer = SimpleImputer(strategy=strategy)
            data[col] = imputer.fit_transform(data[[col]])
            imputers[col] = imputer

    value_mapping = {
        None: 0, "": 0,
        "normal": 1, "present": 1, "yes": 1, "good": 1,
        "abnormal": 2, "notpresent": 2, "no": 2, "poor": 2
    }
    for col in ['rbc', 'pc']:
        data[col] = data[col].map(value_mapping).fillna(0).astype(int)
    for col in ['pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
        data[col] = data[col].fillna(data[col].mode()[0]).map(value_mapping).astype(int)

    logging.info("Mapping target variable to binary values...")
    data['class'] = data['class'].map({"ckd": 1, "notckd": 0})

    logging.info("Splitting data into train and test sets...")
    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logging.info("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    logging.info("Scaling numerical data...")
    scaler = StandardScaler()
    X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    logging.info("Converting target labels to categorical...")
    y_train_categorical = to_categorical(y_train_balanced, num_classes=2)
    y_test_categorical = to_categorical(y_test, num_classes=2)

    logging.info("Preprocessing completed.")
    return data, {
        "X_train_balanced_scaled": X_train_balanced_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_balanced": y_train_balanced,
        "y_test": y_test,
        "y_train_categorical": y_train_categorical,
        "y_test_categorical": y_test_categorical,
        "scaler": scaler,
        "imputers": imputers
    }
