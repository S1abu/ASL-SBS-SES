# src/data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from src import config

def load_dataframe():
    """Scans the dataset directory and returns a DataFrame with paths and labels."""
    all_data = []
    if not os.path.exists(config.DATA_DIR):
        raise FileNotFoundError(f"Dataset not found at {config.DATA_DIR}")

    print("Scanning dataset directories...")
    for folder in os.listdir(config.DATA_DIR):
        label_folder = os.path.join(config.DATA_DIR, folder)
        if os.path.isdir(label_folder):
            files = [
                {'label': folder, 'path': os.path.join(label_folder, f)}
                for f in os.listdir(label_folder)
                if os.path.isfile(os.path.join(label_folder, f))
            ]
            all_data.extend(files)
    
    df = pd.DataFrame(all_data)
    print(f"Found {len(df)} images.")
    return df

def get_generators(df):
    """Splits data and returns Train, Val, and Holdout generators."""
    # Split: Train+Val vs Holdout
    x_train_val, x_holdout = train_test_split(
        df, test_size=config.TEST_SPLIT, random_state=42, stratify=df[['label']]
    )
    
    # Split: Train vs Val
    x_train, x_val = train_test_split(
        x_train_val, test_size=config.VAL_SPLIT, random_state=42, stratify=x_train_val[['label']]
    )

    print(f"Training set: {len(x_train)} images")
    print(f"Validation set: {len(x_val)} images")
    print(f"Holdout set: {len(x_holdout)} images")

    # Generators
    datagen = ImageDataGenerator(rescale=1/255.0)

    train_gen = datagen.flow_from_dataframe(
        dataframe=x_train, x_col='path', y_col='label',
        target_size=config.TARGET_SIZE, class_mode='categorical',
        batch_size=config.BATCH_SIZE, shuffle=True
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=x_val, x_col='path', y_col='label',
        target_size=config.TARGET_SIZE, class_mode='categorical',
        batch_size=config.BATCH_SIZE, shuffle=False
    )

    holdout_gen = datagen.flow_from_dataframe(
        dataframe=x_holdout, x_col='path', y_col='label',
        target_size=config.TARGET_SIZE, class_mode='categorical',
        batch_size=config.BATCH_SIZE, shuffle=False
    )

    return train_gen, val_gen, holdout_gen