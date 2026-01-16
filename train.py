# train.py
import os
import numpy as np
from sklearn import metrics
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from src import config, data, model, utils
import json

def main():
    # 1. Setup GPU
    utils.setup_gpu()

    # 2. Load Data
    df = data.load_dataframe()
    train_gen, val_gen, holdout_gen = data.get_generators(df)
    
    # Save class mapping
    class_mapping_path = os.path.join(config.BASE_DIR, 'class_indices.json')
    with open(class_mapping_path, 'w') as f:
        json.dump(train_gen.class_indices, f)
    print(f"Class mapping saved to {class_mapping_path}")
    
    num_classes = len(train_gen.class_indices)
    print(f"Classes found: {num_classes}")

    # 3. Build Model
    cnn_model = model.build_cnn_model(num_classes)
    cnn_model.summary()

    # 4. Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Starting training...")
    history = cnn_model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        callbacks=[early_stop],
        verbose=1
    )

    # 5. Plot Results
    utils.plot_history(history)

    # 6. Evaluation on Holdout
    print("Evaluating on Holdout Set...")
    predictions = cnn_model.predict(holdout_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=-1)
    true_classes = holdout_gen.classes
    class_labels = list(holdout_gen.class_indices.keys())

    report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # 7. Save Model
    # Fixing the error from your notebook by using .keras extension
    utils.save_model_native(cnn_model, config.MODEL_SAVE_DIR, config.MODEL_NAME)

if __name__ == "__main__":
    main()