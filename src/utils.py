# src/utils.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import keras

def setup_gpu():
    """Configures GPU memory growth to avoid allocation errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Detected: {len(gpus)} device(s) found.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Running on CPU.")

def plot_history(history):
    """Plots accuracy and loss graphs."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(14, 7))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'r', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
    plt.title('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title('Loss')
    plt.legend()
    
    plt.show()

def save_model_native(model, directory, filename):
    """Saves model using the modern Keras 3 format."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename)
    # Using the native keras saving method requested
    try:
        keras.saving.save_model(model, filepath)
        print(f"Model saved successfully at: {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_trained_model(filepath):
    """Loads the compiled Keras model."""
    try:
        model = keras.saving.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def preprocess_image(image_path, target_size):
    """
    Loads and preprocesses a single image for inference.
    1. Reads image
    2. Converts BGR to RGB
    3. Resizes to target size (64, 64)
    4. Normalizes pixel values (0-1)
    5. Adds batch dimension (1, 64, 64, 3)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img