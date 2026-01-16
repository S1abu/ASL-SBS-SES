# src/config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'asl_alphabet_train', 'asl_alphabet_train')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
MODEL_NAME = 'asl_model.keras' # Native Keras format

# Image Params
IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# Hyperparameters
BATCH_SIZE = 128 # Increased from notebook since you have a GPU
EPOCHS = 20
LEARNING_RATE = 0.001
TEST_SPLIT = 0.10
VAL_SPLIT = 0.25 # Of the remaining training data

# Check models directory
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
    print(f"Directory '{MODEL_SAVE_DIR}' created.")
else:
    print(f"Directory '{MODEL_SAVE_DIR}' already exists.")