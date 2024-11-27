import logging
import os
from ckd_predictor import CKDPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base code directory
base_dir = os.path.dirname(__file__)

def main():
    logging.info("Starting CKD Predictor pipeline...")
    # data_path = 'chronic_kidney_disease_full.csv'
    data_path = os.path.join(base_dir, "..", "data", "chronic_kidney_disease_full.csv")
    predictor = CKDPredictor(data_path)

    logging.info("Preprocessing data...")
    predictor.preprocess_data()

    logging.info("Training sklearn models...")
    predictor.train_sklearn_models()

    logging.info("Training neural network models...")
    predictor.train_nn_models()

    logging.info("Evaluating models...")
    results_df = predictor.evaluate_models()
    logging.info(f"Evaluation results:\n{results_df}")

    logging.info("Saving models...")
    predictor.save_models()

    logging.info("Plotting results...")
    predictor.plot_results(results_df)

    logging.info("CKD Predictor pipeline completed.")

if __name__ == "__main__":
    main()
