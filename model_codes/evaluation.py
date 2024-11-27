import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_models(models, nn_models, preprocessed_data):
    logging.info("Starting model evaluation...")
    X_test = preprocessed_data["X_test_scaled"]
    y_test = preprocessed_data["y_test"]

    results = []

    for name, model in models.items():
        logging.info(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        acc_score = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc_score, "AUC": auc_score})

    for name, model in nn_models.items():
        logging.info(f"Evaluating {name}...")
        y_pred = np.argmax(model.predict(X_test[..., None]), axis=1)
        auc_score = roc_auc_score(y_test, model.predict(X_test[..., None])[:, 1])
        acc_score = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc_score, "AUC": auc_score})

    results_df = pd.DataFrame(results).sort_values(by=["Accuracy", "AUC"], ascending=False)
    logging.info("Model evaluation completed.")
    return results_df
