import tensorflow as tf
import numpy as np
import os
import keras
from src import config, data, utils

def representative_data_gen():
    """
    Generates a small subset of real images to help the converter
    calibrate the quantization parameters (min/max values).
    """
    print("Loading data for calibration...")
    # Load data using your existing pipeline
    df = data.load_dataframe()
    # We only need the train generator to get some sample images
    train_gen, _, _ = data.get_generators(df)
    
    # How many samples to use for calibration (100 is usually enough)
    num_calibration_steps = 100
    
    # Iterate through the generator
    count = 0
    for batch_images, _ in train_gen:
        # Loop through images in the batch
        for i in range(batch_images.shape[0]):
            if count >= num_calibration_steps:
                return
            
            # The model expects input shape (1, 64, 64, 3)
            # We take one image and ensure it has the batch dimension
            input_image = np.expand_dims(batch_images[i], axis=0).astype(np.float32)
            
            # Yield the input as a list (required by TFLiteConverter)
            yield [input_image]
            count += 1

def main():
    # 1. Load the Trained Model
    model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)
    print(f"Loading Keras model from: {model_path}")
    
    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
        return

    model = keras.saving.load_model(model_path)

    # 2. Setup TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # --- QUANTIZATION SETTINGS ---
    
    # A. Set Optimization Flag
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # B. Provide Representative Dataset (Crucial for Int8)
    converter.representative_dataset = representative_data_gen
    
    # C. Ensure full integer quantization for all ops (Best for Edge Hardware)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # D. Set Input/Output types to Int8 (Optional: makes inputs strictly int8)
    # If you remove these lines, input/output remain float32 (but weights are int8)
    # Keeping float32 input is often easier for Python inference, 
    # but int8 is better for strict microcontrollers.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("Converting and Quantizing (this may take a moment)...")
    
    try:
        tflite_quant_model = converter.convert()
    except Exception as e:
        print(f"Conversion failed: {e}")
        return

    # 3. Save the Quantized Model
    output_filename = "model_quantized.tflite"
    output_path = os.path.join(config.MODEL_SAVE_DIR, output_filename)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)
        
    print(f"Success! Quantized model saved to: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024:.2f} KB")

if __name__ == "__main__":
    main()