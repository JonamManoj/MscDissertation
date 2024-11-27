# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:54:20 2024

@author: Manoj
"""

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout, SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Ensure reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
plt.rcParams['figure.dpi'] = 100

class CKDPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.ckd_data = pd.read_csv(data_path)
        self.imputers = {}
        self.models = {}
        self.nn_models = {}
        self.model_files_directory = None

    def preprocess_data(self):
        # Step 3: Preprocess the data
        numerical_cols = self.ckd_data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.ckd_data.select_dtypes(include=['object']).columns.tolist()
        if 'class' in categorical_cols:
            categorical_cols.remove('class')

        # Handle missing values for numerical columns
        for col in numerical_cols:
            if self.ckd_data[col].isna().sum() > 0:
                strategy = 'median' if col in ['sg', 'al', 'su', 'bgr', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc'] else 'mean'
                imputer = SimpleImputer(strategy=strategy)
                self.ckd_data[col] = imputer.fit_transform(self.ckd_data[[col]])
                self.imputers[col] = imputer

        # Handle missing values for categorical columns
        value_mapping = {
            None: 0, "": 0,
            "normal": 1, "present": 1, "yes": 1, "good": 1,
            "abnormal": 2, "notpresent": 2, "no": 2, "poor": 2
        }
        for col in ['rbc', 'pc']:
            self.ckd_data[col] = self.ckd_data[col].map(value_mapping).fillna(0).astype(int)
        for col in ['pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
            self.ckd_data[col] = self.ckd_data[col].fillna(self.ckd_data[col].mode()[0]).map(value_mapping).astype(int)

        # Map target class to binary values
        self.ckd_data['class'] = self.ckd_data['class'].map({"ckd": 1, "notckd": 0})

        # Define features and target
        self.X = self.ckd_data.drop('class', axis=1)
        self.y = self.ckd_data['class']

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=RANDOM_STATE, stratify=self.y)

        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=RANDOM_STATE)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)

        # Scale numerical data
        self.scaler = StandardScaler()
        self.X_train_balanced_scaled = self.scaler.fit_transform(self.X_train_balanced)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Convert target to categorical for NN models
        self.y_train_categorical = to_categorical(self.y_train_balanced, num_classes=2)
        self.y_test_categorical = to_categorical(self.y_test, num_classes=2)

        # Reshape data for NN models
        self.X_train_reshaped = self.X_train_balanced_scaled[..., np.newaxis]
        self.X_test_reshaped = self.X_test_scaled[..., np.newaxis]

        # Early stopping
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    def train_sklearn_models(self):
        # Step 4: Train machine learning models
        models = {
            "Logistic Regression": GridSearchCV(LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}, cv=5),
            "SVM": GridSearchCV(SVC(probability=True, random_state=RANDOM_STATE), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}, cv=5),
            "Random Forest": GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}, cv=5),
            "XGBoost": GridSearchCV(xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}, cv=5),
        }
        for name, model in models.items():
            model.fit(self.X_train_balanced_scaled, self.y_train_balanced)
        self.models = models

    def train_nn_models(self):
        # Train TensorFlow NN models
        def create_nn_model(input_shape, architecture):
            model = tf.keras.Sequential()
            for layer in architecture:
                model.add(layer)
            model.add(Dense(2, activation='softmax'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        nn_models = {
            "ANN": create_nn_model(self.X_train_balanced_scaled.shape[1:], [Dense(128, activation='relu'), Dropout(0.2), Dense(64, activation='relu')]),
            "CNN": create_nn_model(self.X_train_reshaped.shape[1:], [Conv1D(32, 3, activation='relu'), Conv1D(64, 3, activation='relu'), Flatten()]),
            "RNN": create_nn_model(self.X_train_reshaped.shape[1:], [SimpleRNN(64, activation='relu', return_sequences=True), SimpleRNN(32, activation='relu')]),
            "LSTM": create_nn_model(self.X_train_reshaped.shape[1:], [LSTM(64, activation='tanh', return_sequences=True), LSTM(32, activation='tanh')]),
        }

        for name, model in nn_models.items():
            input_data = self.X_train_balanced_scaled if "ANN" in name else self.X_train_reshaped
            model.fit(input_data, self.y_train_categorical, validation_data=(self.X_test_scaled, self.y_test_categorical),
                      epochs=50, batch_size=32, callbacks=[self.early_stopping], verbose=1)
        self.nn_models = nn_models

    def evaluate_models(self):
        # Combine and evaluate all models
        results = []
        for name, model in self.models.items():
            y_pred = model.best_estimator_.predict(self.X_test_scaled)
            auc_score = roc_auc_score(self.y_test, y_pred)
            acc_score = accuracy_score(self.y_test, y_pred)
            results.append({"Model": name, "Accuracy": acc_score, "AUC": auc_score})

        for name, model in self.nn_models.items():
            input_data = self.X_test_scaled if "ANN" in name else self.X_test_reshaped
            y_pred_proba = model.predict(input_data)
            y_pred = np.argmax(y_pred_proba, axis=1)
            auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            acc_score = accuracy_score(self.y_test, y_pred)
            results.append({"Model": name, "Accuracy": acc_score, "AUC": auc_score})

        # Convert results to DataFrame
        results_df = pd.DataFrame(results).sort_values(by=["Accuracy", "AUC"], ascending=False)
        return results_df

    def save_models(self):
        # Step 5: Save all models and preprocessing objects
        self.model_files_directory = os.path.join(os.path.dirname(self.data_path), "model_files")
        os.makedirs(self.model_files_directory, exist_ok=True)

        # Save preprocessing objects
        for col, imputer in self.imputers.items():
            joblib.dump(imputer, os.path.join(self.model_files_directory, f"imputer_{col}.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_files_directory, "scaler.pkl"))

        # Save sklearn models
        for name, model in self.models.items():
            joblib.dump(model.best_estimator_, os.path.join(self.model_files_directory, f"{name.replace(' ', '_').lower()}_model.pkl"))

        # Save NN models
        for name, model in self.nn_models.items():
            model.save(os.path.join(self.model_files_directory, f"{name.lower()}_model.keras"))

        print(f"All models and preprocessing objects are saved in the directory: {self.model_files_directory}")

    def plot_results(self, results_df):
        # Step 6: Visualizations
        # ROC Curves
        plt.figure(figsize=(10, 8))

        for name, model in self.models.items():
            if hasattr(model.best_estimator_, "predict_proba"):
                y_prob = model.best_estimator_.predict_proba(self.X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

        for name, model in self.nn_models.items():
            input_data = self.X_test_scaled if "ANN" in name else self.X_test_reshaped
            y_prob = model.predict(input_data)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multiple Models')
        plt.legend(loc="lower right")
        plt.show()

        # Accuracy Bar Plot
        plt.figure(figsize=(10, 6))
        plt.barh(results_df["Model"], results_df["Accuracy"], color='skyblue')
        plt.xlabel('Accuracy')
        plt.title('Accuracy of Different Models')
        plt.xlim(0.0, 1.0)
        plt.show()

        # Confusion Matrices
        for name, model in self.models.items():
            y_pred = model.best_estimator_.predict(self.X_test_scaled)
            disp = ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred)
            disp.ax_.set_title(f'Confusion Matrix for {name}')
            plt.show()

        for name, model in self.nn_models.items():
            input_data = self.X_test_scaled if "ANN" in name else self.X_test_reshaped
            y_pred = np.argmax(model.predict(input_data), axis=1)
            disp = ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred)
            disp.ax_.set_title(f'Confusion Matrix for {name}')
            plt.show()


def main():
    data_path = 'chronic_kidney_disease_full.csv'
    predictor = CKDPredictor(data_path)

    predictor.preprocess_data()
    predictor.train_sklearn_models()
    predictor.train_nn_models()

    results_df = predictor.evaluate_models()
    print(results_df)

    predictor.save_models()
    predictor.plot_results(results_df)


if __name__ == "__main__":
    main()
