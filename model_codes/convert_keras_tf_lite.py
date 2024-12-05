import tensorflow as tf
from keras.config import enable_unsafe_deserialization

def convert_keras_to_tflite(keras_model_path, output_tflite_path):
    """Convert Keras model to TensorFlow Lite format."""
    # Enable unsafe deserialization
    enable_unsafe_deserialization()

    # Load the Keras model
    model = tf.keras.models.load_model(keras_model_path, safe_mode=False)

    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the converted TFLite model
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model successfully converted to {output_tflite_path}")

if __name__ == "__main__":
    keras_model_path = "model_training_output_files/best_model.keras"  # Path to your .keras file
    output_tflite_path = "model_training_output_files/best_model.tflite"  # Desired .tflite file path
    convert_keras_to_tflite(keras_model_path, output_tflite_path)
