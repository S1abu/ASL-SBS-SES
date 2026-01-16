# src/model.py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam
from src import config

def build_cnn_model(num_classes):
    """Defines the CNN architecture."""
    model = Sequential([
        Input(shape=(config.IMG_WIDTH, config.IMG_HEIGHT, config.CHANNELS)),
        
        Conv2D(32, (5, 5), padding='Same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(64, (5, 5), padding='Same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(64, (5, 5), padding='Same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(128, (5, 5), padding='Same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model