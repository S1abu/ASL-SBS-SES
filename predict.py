# predict.py
import os
import numpy as np
from src import config, utils
import json
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str,
                help="path to image")
args = vars(ap.parse_args())

# Load mapping
mapping_path = os.path.join(config.BASE_DIR, 'class_indices.json')
with open(mapping_path, 'r') as f:
    indices = json.load(f)
    # Swap keys and values: {'A': 0} -> {0: 'A'}
    CLASS_LABELS = {v: k for k, v in indices.items()}

"""
# Define your class labels explicitly in alphabetical order
# (This matches the default behavior of Keras ImageDataGenerator)
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'del', 'nothing', 'space'
]
"""
def predict_custom_image(image_path):
    # 1. Load Model
    predicted_label = CLASS_LABELS[predicted_index]
    model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)
    model = utils.load_trained_model(model_path)
    
    if model is None:
        return

    # 2. Preprocess Image
    try:
        processed_img = utils.preprocess_image(image_path, config.TARGET_SIZE)
    except FileNotFoundError as e:
        print(e)
        return

    # 3. Predict
    print(f"🔍 Analyzing image: {image_path}...")
    predictions = model.predict(processed_img, verbose=0)
    
    # 4. Decode Result
    predicted_index = np.argmax(predictions, axis=-1)[0]
    confidence = np.max(predictions)
    predicted_label = CLASS_LABELS[predicted_index]

    print(f"\nPrediction: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    
    return predicted_label

if __name__ == "__main__":
    predict_custom_image(args["image"])