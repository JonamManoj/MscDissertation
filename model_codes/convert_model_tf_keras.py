import os
import joblib
import tensorflow as tf
import numpy as np

def detect_and_convert_model(model_path):
    """Detect the extension of the model and convert accordingly."""
    _, extension = os.path.splitext(model_path)

    if extension == ".pkl":
        print(f"Detected {extension}: Converting Scikit-learn model to TensorFlow.")
        model = joblib.load(model_path)
        
        if hasattr(model, "kernel") and model.kernel != "linear":
            print("Non-linear kernel detected. Approximating decision boundary.")

            # Extract support vectors, dual coefficients, and intercepts
            support_vectors = model.support_vectors_
            dual_coef = model.dual_coef_
            intercept = model.intercept_

            # Create a custom TensorFlow model
            def svm_decision_function(x):
                """TensorFlow approximation of the SVM decision function."""
                kernel = tf.matmul(x, support_vectors.T)
                return tf.matmul(kernel, dual_coef.T) + intercept

            input_dim = support_vectors.shape[1]
            tf_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Lambda(lambda x: svm_decision_function(x))
            ])
        else:
            print("Linear kernel detected. Converting weights.")
            tf_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(model.coef_.shape[1],)),
                tf.keras.layers.Dense(1, activation="sigmoid",
                                      weights=[model.coef_.T, model.intercept_])
            ])

        # Save the TensorFlow model
        keras_path = model_path.replace(".pkl", ".keras")
        tf_model.save(keras_path)
        print(f"Model converted to {keras_path}")
        return keras_path

    elif extension == ".keras":
        print("Detected .keras model. No conversion needed.")
        return model_path

    else:
        raise ValueError("Unsupported model format. Please use .pkl or .keras.")

if __name__ == "__main__":
    model_path = "model_training_output_files/best_model.pkl"  # Update this to the actual model path
    converted_model_path = detect_and_convert_model(model_path)
    print(f"Converted model is available at {converted_model_path}")

