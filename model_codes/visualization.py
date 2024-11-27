import logging
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
import numpy as np

def plot_results(models, nn_models, preprocessed_data, results_df, model_files_directory):
    logging.info("Generating and saving visualizations...")
    X_test = preprocessed_data["X_test_scaled"]
    y_test = preprocessed_data["y_test"]

    # Directory to save plots
    plots_dir = os.path.join(model_files_directory, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot ROC Curves
    roc_plot_path = os.path.join(plots_dir, "roc_curves.png")
    plt.figure(figsize=(12, 8))

    # Sklearn models
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

    # NN models
    for name, model in nn_models.items():
        logging.info(f"Generating ROC curve for {name}...")
        y_prob = model.predict(X_test[..., None])[:, 1]  # Extract probabilities for positive class
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(roc_plot_path)
    plt.close()
    logging.info(f"ROC curves saved to {roc_plot_path}")

    # Accuracy Bar Plot
    accuracy_plot_path = os.path.join(plots_dir, "accuracy_bar_plot.png")
    logging.info("Generating accuracy bar plot...")
    plt.figure(figsize=(10, 6))
    plt.barh(results_df["Model"], results_df["Accuracy"], color='skyblue')
    plt.xlabel("Accuracy")
    plt.title("Model Accuracies")
    plt.gca().invert_yaxis()
    plt.savefig(accuracy_plot_path)
    plt.close()
    logging.info(f"Accuracy bar plot saved to {accuracy_plot_path}")

    # Confusion Matrices
    for name, model in models.items():
        logging.info(f"Generating confusion matrix for {name}...")
        y_pred = model.predict(X_test)
        confusion_matrix_path = os.path.join(plots_dir, f"confusion_matrix_{name}.png")
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        disp.figure_.suptitle(f'Confusion Matrix - {name}')
        disp.figure_.savefig(confusion_matrix_path)
        plt.close(disp.figure_)
        logging.info(f"Confusion matrix for {name} saved to {confusion_matrix_path}")

    for name, model in nn_models.items():
        logging.info(f"Generating confusion matrix for {name}...")
        y_pred = np.argmax(model.predict(X_test[..., None]), axis=1)
        confusion_matrix_path = os.path.join(plots_dir, f"confusion_matrix_{name}.png")
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        disp.figure_.suptitle(f'Confusion Matrix - {name}')
        disp.figure_.savefig(confusion_matrix_path)
        plt.close(disp.figure_)
        logging.info(f"Confusion matrix for {name} saved to {confusion_matrix_path}")

    logging.info("All visualizations have been generated and saved.")
