# -*- coding: utf-8 -*-
"""
CKD Predictor with Embedded Preprocessing, Model Saving, and Visualizations
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Flatten, LSTM, SimpleRNN, Normalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


class CKDPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.ckd_data = pd.read_csv(data_path)
        self.models = {}
        self.tflite_models_directory = None
        self.visualization_directory = None

    def preprocess_data(self):
        # Handle missing values and map categorical features
        numerical_cols = self.ckd_data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.ckd_data.select_dtypes(include=['object']).columns.tolist()
        if 'class' in categorical_cols:
            categorical_cols.remove('class')

        for col in numerical_cols:
            strategy = 'median' if col in ['sg', 'al', 'su', 'bgr', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc'] else 'mean'
            imputer = SimpleImputer(strategy=strategy)
            self.ckd_data[col] = imputer.fit_transform(self.ckd_data[[col]])

        value_mapping = {
            None: 0, "": 0, "normal": 1, "present": 1, "yes": 1, "good": 1,
            "abnormal": 2, "notpresent": 2, "no": 2, "poor": 2
        }
        for col in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
            self.ckd_data[col] = self.ckd_data[col].map(value_mapping).fillna(0).astype(int)

        self.ckd_data['class'] = self.ckd_data['class'].map({"ckd": 1, "notckd": 0})

        # Split features and target
        self.X = self.ckd_data.drop('class', axis=1)
        self.y = self.ckd_data['class']

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Balance classes with SMOTE
        smote = SMOTE(random_state=42)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)

        # Convert X_train_balanced to NumPy array
        self.X_train_balanced = self.X_train_balanced.to_numpy()

        # Preprocessing layers
        self.normalization_layer = Normalization()
        self.normalization_layer.adapt(self.X_train_balanced)


    def create_model(self, input_shape, architecture):
        inputs = Input(shape=input_shape)
        x = inputs

        # Reshape input for CNN, RNN, and LSTM models
        if any(isinstance(layer, Conv1D) or isinstance(layer, SimpleRNN) or isinstance(layer, LSTM) for layer in architecture):
            x = tf.keras.layers.Reshape((input_shape[0], 1))(x)  # Add 1 channel dimension

        # Add architecture layers
        for layer in architecture:
            x = layer(x)

        outputs = Dense(2, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def train_models(self):
        # Define architectures for 8 models
        self.models = {
            "Logistic Regression (Keras)": self.create_model(
                self.X_train_balanced.shape[1:], [Dense(1, activation='sigmoid')]
            ),
            "ANN": self.create_model(
                self.X_train_balanced.shape[1:], [Dense(128, activation='relu'), Dropout(0.2), Dense(64, activation='relu')]
            ),
            "CNN": self.create_model(
                (self.X_train_balanced.shape[1], 1), [Conv1D(32, 3, activation='relu'), Conv1D(64, 3, activation='relu'), Flatten()]
            ),
            "RNN": self.create_model(
                (self.X_train_balanced.shape[1], 1), [SimpleRNN(64, activation='relu', return_sequences=True), SimpleRNN(32, activation='relu')]
            ),
            "LSTM": self.create_model(
                (self.X_train_balanced.shape[1], 1), [LSTM(64, activation='tanh', return_sequences=True), LSTM(32, activation='tanh')]
            ),
            "SVM (Approximation)": self.create_model(
                self.X_train_balanced.shape[1:], [Dense(64, activation='linear')]
            ),
            "XGBoost (Approximation)": self.create_model(
                self.X_train_balanced.shape[1:], [Dense(128, activation='relu'), Dense(64, activation='relu'), Dense(32, activation='relu')]
            ),
            "Decision Tree (Approximation)": self.create_model(
                self.X_train_balanced.shape[1:], [Dense(64, activation='relu'), Dropout(0.2), Dense(32, activation='relu')]
            ),
        }

        # Train models
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(
                self.X_train_balanced, tf.keras.utils.to_categorical(self.y_train_balanced, num_classes=2),
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                verbose=1,
            )

    def save_models(self):
        # Save models as TFLite
        self.tflite_models_directory = os.path.join("model_training_output_files", "models")
        os.makedirs(self.tflite_models_directory, exist_ok=True)

        for name, model in self.models.items():
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            tflite_path = os.path.join(self.tflite_models_directory, f"{name.lower().replace(' ', '_')}.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            print(f"Saved TFLite model for {name}: {tflite_path}")

    def evaluate_and_visualize_models(self):
        self.visualization_directory = os.path.join("model_training_output_files", "visualizations")
        os.makedirs(self.visualization_directory, exist_ok=True)

        results = []
        plt.figure(figsize=(10, 8))

        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            predictions = model.predict(self.X_test)
            predicted_classes = np.argmax(predictions, axis=1)
            acc = accuracy_score(self.y_test, predicted_classes)
            auc_score = roc_auc_score(self.y_test, predictions[:, 1])
            print(f"{name}: Accuracy = {acc:.4f}, AUC = {auc_score:.4f}")
            results.append({"Model": name, "Accuracy": acc, "AUC": auc_score})

            # Plot ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, predictions[:, 1])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

        # Save ROC Comparison
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.visualization_directory, "roc_comparison.png"))
        plt.close()

        # Save Accuracy Comparison
        results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        plt.figure(figsize=(10, 6))
        plt.barh(results_df["Model"], results_df["Accuracy"], color='skyblue')
        plt.xlabel('Accuracy')
        plt.title('Accuracy Comparison')
        plt.savefig(os.path.join(self.visualization_directory, "accuracy_comparison.png"))
        plt.close()

        print("\nFinal Model Evaluation Results:")
        print(results_df)

        # Save results to CSV
        results_df.to_csv(os.path.join(self.visualization_directory, "model_comparison.csv"), index=False)


def main():
    data_path = 'data/chronic_kidney_disease_full.csv'
    predictor = CKDPredictor(data_path)

    predictor.preprocess_data()
    predictor.train_models()
    predictor.evaluate_and_visualize_models()
    predictor.save_models()


if __name__ == "__main__":
    main()
