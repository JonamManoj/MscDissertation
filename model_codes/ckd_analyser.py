import os
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout, SimpleRNN, Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay
import tensorflow as tf

# Configure logging to display progress and debugging information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CKDAnalyser:
    def __init__(self, data_path):
        """
        Initializes the CKDAnalyser class with data path and directories for saving models and plots.
        Parameters:
            data_path (str): The file path to the dataset.
        """
        logging.info("Initializing CKDAnalyser...")
        self.data_path = data_path
        self.ckd_data = None
        self.models = {}
        self.nn_models = {}
        self.preprocessed_data = None
        self.model_files_directory = "model_training_output_files"
        self.plots_directory = os.path.join(self.model_files_directory, "plots")
        os.makedirs(self.plots_directory, exist_ok=True)

    def preprocess_data(self):
        """
        Load and preprocess the dataset, including handling missing values, encoding categorical features,
        splitting the dataset, applying SMOTE for class balancing, and scaling numerical features.
        """
        logging.info("Loading data from CSV...")
        data = pd.read_csv(self.data_path)

        # Handle missing values and encode categorical columns
        logging.info("Handling missing values and encoding categorical columns...")
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if 'class' in categorical_cols:
            categorical_cols.remove('class')

        # Impute missing values for numerical columns
        imputers = {}
        for col in numerical_cols:
            if data[col].isna().sum() > 0:
                strategy = 'median' if col in ['sg', 'al', 'su', 'bgr', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc'] else 'mean'
                imputer = SimpleImputer(strategy=strategy)
                data[col] = imputer.fit_transform(data[[col]])
                imputers[col] = imputer

        # Map categorical values to numerical values
        value_mapping = {
            None: 0, "": 0,
            "normal": 1, "present": 1, "yes": 1, "good": 1,
            "abnormal": 2, "notpresent": 2, "no": 2, "poor": 2
        }
        for col in ['rbc', 'pc']:
            data[col] = data[col].map(value_mapping).fillna(0).astype(int)
        for col in ['pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
            data[col] = data[col].fillna(data[col].mode()[0]).map(value_mapping).astype(int)

        # Convert target variable to binary values
        logging.info("Mapping target variable to binary values...")
        data['class'] = data['class'].map({"ckd": 1, "notckd": 0})

        # Split the data into training and testing sets
        logging.info("Splitting data into train and test sets...")
        X = data.drop('class', axis=1)
        y = data['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Apply SMOTE to balance the classes
        logging.info("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Scale the numerical data
        logging.info("Scaling numerical data...")
        scaler = StandardScaler()
        X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)

        # Convert target labels to categorical format for neural network models
        logging.info("Converting target labels to categorical...")
        y_train_categorical = to_categorical(y_train_balanced, num_classes=2)
        y_test_categorical = to_categorical(y_test, num_classes=2)

        logging.info("Preprocessing completed.")
        # Store preprocessed data in the instance variable
        self.preprocessed_data = {
            "X_train_balanced_scaled": X_train_balanced_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train_balanced": y_train_balanced,
            "y_test": y_test,
            "y_train_categorical": y_train_categorical,
            "y_test_categorical": y_test_categorical,
            "scaler": scaler,
            "imputers": imputers
        }

    def train_sklearn_models(self):
        """
        Train traditional machine learning models using sklearn with hyperparameter tuning using GridSearchCV.
        The models trained are Logistic Regression, SVM, Random Forest, and XGBoost.
        """
        logging.info("Training sklearn models...")
        X_train = self.preprocessed_data["X_train_balanced_scaled"]
        y_train = self.preprocessed_data["y_train_balanced"]

        # Define models with more advanced hyperparameter tuning using GridSearchCV
        self.models = {
            "Logistic Regression": GridSearchCV(LogisticRegression(max_iter=5000), {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}),
            "SVM": GridSearchCV(SVC(probability=True), {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}),
            "Random Forest": GridSearchCV(RandomForestClassifier(), {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}),
            "XGBoost": GridSearchCV(xgb.XGBClassifier(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 9]})
        }

        # Train each model
        for name, model in self.models.items():
            logging.info(f"Training {name}...")
            model.fit(X_train, y_train)

        logging.info("Sklearn models training completed.")

    def train_nn_models(self):
        """
        Train neural network models using Keras. The models trained include ANN, CNN, RNN, and LSTM.
        """
        logging.info("Training neural network models...")
        X_train = self.preprocessed_data["X_train_balanced_scaled"]
        y_train = self.preprocessed_data["y_train_categorical"]

        # Define early stopping callback to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Define neural network architectures
        self.nn_models = {
            "ANN": Sequential([
                Input(shape=(X_train.shape[1],)),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(2, activation='softmax')
            ]),
            "CNN": Sequential([
                Input(shape=(X_train.shape[1], 1)),
                Conv1D(32, 3, activation='relu'),
                Conv1D(64, 3, activation='relu'),
                Flatten(),
                Dense(2, activation='softmax')
            ]),
            "RNN": Sequential([
                Input(shape=(X_train.shape[1], 1)),
                SimpleRNN(64, activation='relu', return_sequences=True),
                SimpleRNN(32, activation='relu'),
                Dense(2, activation='softmax')
            ]),
            "LSTM": Sequential([
                Input(shape=(X_train.shape[1], 1)),
                LSTM(64, activation='tanh', return_sequences=True),
                LSTM(32, activation='tanh'),
                Dense(2, activation='softmax')
            ])
        }

        # Train each neural network model
        for name, model in self.nn_models.items():
            logging.info(f"Training {name}...")
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train[..., None], y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

        logging.info("Neural network models training completed.")

    def evaluate_models(self):
        """
        Evaluate all trained models on the test set and return the results as a DataFrame.
        Both sklearn and neural network models are evaluated based on Accuracy and AUC score.
        """
        logging.info("Evaluating models...")
        X_test = self.preprocessed_data["X_test_scaled"]
        y_test = self.preprocessed_data["y_test"]

        results = []

        # Evaluate sklearn models
        for name, model in self.models.items():
            logging.info(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            acc_score = accuracy_score(y_test, y_pred)
            results.append({"Model": name, "Accuracy": acc_score, "AUC": auc_score})

        # Evaluate neural network models
        for name, model in self.nn_models.items():
            logging.info(f"Evaluating {name}...")
            y_pred = np.argmax(model.predict(X_test[..., None]), axis=1)
            auc_score = roc_auc_score(y_test, model.predict(X_test[..., None])[:, 1])
            acc_score = accuracy_score(y_test, y_pred)
            results.append({"Model": name, "Accuracy": acc_score, "AUC": auc_score})

        # Create a DataFrame to store results and sort by accuracy and AUC
        results_df = pd.DataFrame(results).sort_values(by=["Accuracy", "AUC"], ascending=False)
        logging.info("Model evaluation completed.")
        return results_df

    def save_models(self):
        """
        Save all trained models and preprocessing artifacts, including sklearn models, neural network models,
        scaler, and imputers.
        """
        logging.info("Saving trained models and preprocessing artifacts...")
        os.makedirs(self.model_files_directory, exist_ok=True)

        # Save Sklearn models
        for name, model in self.models.items():
            model_file_path = os.path.join(self.model_files_directory, f"{name}_model.pkl")
            joblib.dump(model, model_file_path)
            logging.info(f"Saved Sklearn model: {name} at {model_file_path}")

        # Save Neural Network models
        for name, model in self.nn_models.items():
            model_file_path = os.path.join(self.model_files_directory, f"{name}_model.keras")
            model.save(model_file_path)
            logging.info(f"Saved Neural Network model: {name} at {model_file_path}")

        # Save preprocessing artifacts
        preprocessing_dir = os.path.join(self.model_files_directory, "preprocessing")
        os.makedirs(preprocessing_dir, exist_ok=True)

        # Save scaler
        scaler_path = os.path.join(preprocessing_dir, "scaler.pkl")
        joblib.dump(self.preprocessed_data["scaler"], scaler_path)
        logging.info(f"Saved scaler at {scaler_path}")

        # Save imputers
        imputers_path = os.path.join(preprocessing_dir, "imputers.pkl")
        joblib.dump(self.preprocessed_data["imputers"], imputers_path)
        logging.info(f"Saved imputers at {imputers_path}")

        logging.info("All models and preprocessing artifacts saved successfully.")

    def plot_results(self, results_df):
        """
        Generate and save visualizations for model evaluation, including ROC curves, accuracy bar plots,
        and confusion matrices.
        """
        logging.info("Generating and saving visualizations...")
        X_test = self.preprocessed_data["X_test_scaled"]
        y_test = self.preprocessed_data["y_test"]

        # Plot ROC Curves
        roc_plot_path = os.path.join(self.plots_directory, "roc_curves.png")
        plt.figure(figsize=(12, 8))

        # Plot ROC curve for sklearn models
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

        # Plot ROC curve for neural network models
        for name, model in self.nn_models.items():
            logging.info(f"Generating ROC curve for {name}...")
            y_prob = model.predict(X_test[..., None])[:, 1]  # Extract probabilities for positive class
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

        # Plot the diagonal line for reference
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig(roc_plot_path)
        plt.close()
        logging.info(f"ROC curves saved to {roc_plot_path}")

        # Improved Accuracy Bar Plot using Seaborn for better visualization
        accuracy_plot_path = os.path.join(self.plots_directory, "accuracy_bar_plot.png")
        logging.info("Generating improved accuracy bar plot...")
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis")
        plt.xlabel("Accuracy")
        plt.title("Model Accuracies")
        plt.tight_layout()
        plt.savefig(accuracy_plot_path)
        plt.close()
        logging.info(f"Improved accuracy bar plot saved to {accuracy_plot_path}")

        # Generate confusion matrices for each model
        for name, model in self.models.items():
            logging.info(f"Generating confusion matrix for {name}...")
            y_pred = model.predict(X_test)
            confusion_matrix_path = os.path.join(self.plots_directory, f"confusion_matrix_{name}.png")
            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
            disp.figure_.suptitle(f'Confusion Matrix - {name}')
            disp.figure_.savefig(confusion_matrix_path)
            plt.close(disp.figure_)
            logging.info(f"Confusion matrix for {name} saved to {confusion_matrix_path}")

        for name, model in self.nn_models.items():
            logging.info(f"Generating confusion matrix for {name}...")
            y_pred = np.argmax(model.predict(X_test[..., None]), axis=1)
            confusion_matrix_path = os.path.join(self.plots_directory, f"confusion_matrix_{name}.png")
            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
            disp.figure_.suptitle(f'Confusion Matrix - {name}')
            disp.figure_.savefig(confusion_matrix_path)
            plt.close(disp.figure_)
            logging.info(f"Confusion matrix for {name} saved to {confusion_matrix_path}")

        logging.info("All visualizations have been generated and saved.")

    def run_pipeline(self):
        """
        Run the complete CKD analysis pipeline including data preprocessing, model training, model evaluation,
        saving models, and generating visualizations.
        """
        logging.info("Starting CKD Analyser pipeline...")
        self.preprocess_data()
        self.train_sklearn_models()
        self.train_nn_models()
        results_df = self.evaluate_models()
        logging.info(f"Evaluation results:\n{results_df}")
        self.save_models()
        self.plot_results(results_df)
        logging.info("CKD Analyser pipeline completed.")

if __name__ == "__main__":
    # Define base directory and data path, then run the pipeline
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "chronic_kidney_disease_full.csv")
    analyser = CKDAnalyser(data_path)
    analyser.run_pipeline()
