# convert_to_tflite.py
import tensorflow as tf
import os
import keras
from src import config

def main():
    # 1. Path to your trained .keras model
    model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)
    
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print("Model not found! Run train.py first.")
        return

    # Load the Keras model
    model = keras.saving.load_model(model_path)

    # 2. Convert to TensorFlow Lite
    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # 3. Save the .tflite file
    tflite_filename = "model.tflite"
    tflite_path = os.path.join(config.MODEL_SAVE_DIR, tflite_filename)
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Success! Model saved to: {tflite_path}")
    print("Upload this 'model.tflite' file to Edge Impulse.")

if __name__ == "__main__":
    main()