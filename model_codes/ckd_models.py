# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# Ensure reproducibility and improve plots' resolution
RANDOM_STATE = 42
plt.rcParams['figure.dpi'] = 100

# Step 2: Load the dataset
data_path = 'chronic_kidney_disease_full.csv'
data_directory = os.path.dirname(data_path)
ckd_data = pd.read_csv(data_path)

# Step 3: Preprocess the data
numerical_cols = ckd_data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = ckd_data.select_dtypes(include=['object']).columns.tolist()
if 'class' in categorical_cols:
    categorical_cols.remove('class')

# Handle missing values for numerical columns
imputers = {}
for col in numerical_cols:
    if ckd_data[col].isna().sum() > 0:
        if col in ['sg', 'al', 'su', 'bgr', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']:
            imputer = SimpleImputer(strategy='median')
        else:
            imputer = SimpleImputer(strategy='mean')
        ckd_data[col] = imputer.fit_transform(ckd_data[[col]])
        imputers[col] = imputer

# Handle missing values for categorical columns
# Apply custom mapping and mode imputation
value_mapping = {
    None: 0, "": 0,
    "normal": 1, "present": 1, "yes": 1, "good": 1,
    "abnormal": 2, "notpresent": 2, "no": 2, "poor": 2
}

# Apply mapping for custom categorical columns
for col in ['rbc', 'pc']:
    ckd_data[col] = ckd_data[col].map(value_mapping).fillna(0).astype(int)

# Handle missing values for other categorical columns using Mode Imputation
for col in ['pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
    ckd_data[col] = ckd_data[col].fillna(ckd_data[col].mode()[0]).map(value_mapping).astype(int)

# Map target class to binary values
ckd_data['class'] = ckd_data['class'].map({"ckd": 1, "notckd": 0})

# Define features and target
X = ckd_data.drop('class', axis=1)
y = ckd_data['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale numerical data
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train models with hyperparameter tuning
# Logistic Regression
logistic_params = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
logistic_model = GridSearchCV(LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), logistic_params, cv=5)
logistic_model.fit(X_train_balanced_scaled, y_train_balanced)

# KNN
knn_params = {'n_neighbors': [3, 5, 7]}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_model.fit(X_train_balanced_scaled, y_train_balanced)

# SVM
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_model = GridSearchCV(SVC(probability=True, random_state=RANDOM_STATE), svm_params, cv=5)
svm_model.fit(X_train_balanced_scaled, y_train_balanced)

# Random Forest
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf_model = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), rf_params, cv=5)
rf_model.fit(X_train_balanced, y_train_balanced)

# XGBoost
xgb_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
xgb_model = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE), xgb_params, cv=5)
xgb_model.fit(X_train_balanced, y_train_balanced)

# MLP
mlp_params = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001]}
mlp_model = GridSearchCV(MLPClassifier(max_iter=1000, random_state=RANDOM_STATE), mlp_params, cv=5)
mlp_model.fit(X_train_balanced_scaled, y_train_balanced)

# Step 5: Evaluate models dynamically
model_metrics = {}
for model_name, model in {
    "Logistic Regression": logistic_model,
    "KNN": knn_model,
    "SVM": svm_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "MLP (Neural Network)": mlp_model
}.items():
    if hasattr(model.best_estimator_, "predict_proba"):
        y_prob = model.best_estimator_.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.best_estimator_.predict(X_test_scaled)
        model_metrics[model_name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob)
        }
    else:
        y_pred = model.best_estimator_.predict(X_test_scaled)
        model_metrics[model_name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": None  # AUC not available for models without predict_proba
        }

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(model_metrics).T
metrics_df.reset_index(inplace=True)
metrics_df.columns = ["Model", "Accuracy", "AUC"]
metrics_df.sort_values(by=["Accuracy", "AUC"], ascending=False, inplace=True)

print(metrics_df)

# Identify the best and worst models
best_model_name = metrics_df.iloc[0]["Model"]
worst_model_name = metrics_df.iloc[-1]["Model"]

model_map = {
    "Logistic Regression": logistic_model,
    "KNN": knn_model,
    "SVM": svm_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "MLP (Neural Network)": mlp_model
}

best_model = model_map[best_model_name].best_estimator_
worst_model = model_map[worst_model_name].best_estimator_

# Evaluation report for the best model
print(f"Classification Report for {best_model_name}:")
y_pred_best = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best))

# Create the 'model_files' directory if it doesn't exist
model_files_directory = os.path.join(data_directory, "model_files")
os.makedirs(model_files_directory, exist_ok=True)

# Save preprocessing objects
for col, imputer in imputers.items():
    joblib.dump(imputer, os.path.join(model_files_directory, f"imputer_{col}.pkl"))
joblib.dump(scaler, os.path.join(model_files_directory, "scaler.pkl"))

# Save each model to a separate file
for model_name, model in model_map.items():
    joblib.dump(model.best_estimator_, os.path.join(model_files_directory, f"{model_name.replace(' ', '_').lower()}_model.pkl"))

# Save the best and worst models separately for easy access
joblib.dump(best_model, os.path.join(model_files_directory, "best_ckd_model.pkl"))
joblib.dump(worst_model, os.path.join(model_files_directory, "worst_ckd_model.pkl"))

print(f"All models and preprocessing objects are saved in the directory: {model_files_directory}")

# Step 8: Visualizations
# ROC Curves
plt.figure(figsize=(10, 8))

for model_name, model in model_map.items():
    if hasattr(model.best_estimator_, "predict_proba"):
        y_prob = model.best_estimator_.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multiple Models')
plt.legend(loc="lower right")
plt.show()

# Accuracy Bar Plot
plt.figure(figsize=(10, 6))
plt.barh(metrics_df["Model"], metrics_df["Accuracy"], color='skyblue')
plt.xlabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.xlim(0.0, 1.0)
plt.show()

# Confusion Matrices
for model_name, model in model_map.items():
    y_pred = model.best_estimator_.predict(X_test_scaled)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.ax_.set_title(f'Confusion Matrix for {model_name}')
    plt.show()
