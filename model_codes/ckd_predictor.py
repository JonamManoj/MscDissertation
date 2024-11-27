import logging
from preprocessing import preprocess_data
from model_training import train_sklearn_models, train_nn_models
from evaluation import evaluate_models
from utils import save_models
from visualization import plot_results

class CKDPredictor:
    def __init__(self, data_path):
        logging.info("Initializing CKDPredictor...")
        self.data_path = data_path
        self.ckd_data = None
        self.models = {}
        self.nn_models = {}
        self.preprocessed_data = None
        self.model_files_directory = "model_training_output_files"

    def preprocess_data(self):
        logging.info("Preprocessing data...")
        self.ckd_data, self.preprocessed_data = preprocess_data(self.data_path)
        logging.info("Data preprocessing completed.")

    def train_sklearn_models(self):
        logging.info("Training sklearn models...")
        self.models = train_sklearn_models(self.preprocessed_data)
        logging.info("Sklearn models training completed.")

    def train_nn_models(self):
        logging.info("Training neural network models...")
        self.nn_models = train_nn_models(self.preprocessed_data)
        logging.info("Neural network models training completed.")

    def evaluate_models(self):
        logging.info("Evaluating models...")
        results = evaluate_models(self.models, self.nn_models, self.preprocessed_data)
        logging.info("Model evaluation completed.")
        return results

    def save_models(self):
        logging.info("Saving trained models and preprocessing artifacts...")
        save_models(self.models, self.nn_models, self.preprocessed_data, self.model_files_directory)
        logging.info("Models and preprocessing artifacts saved.")

    def plot_results(self, results_df):
        logging.info("Generating visualizations...")
        plot_results(self.models, self.nn_models, self.preprocessed_data, results_df, self.model_files_directory)
        logging.info("Visualizations generated.")
