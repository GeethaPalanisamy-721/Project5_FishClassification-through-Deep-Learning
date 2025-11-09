"""
train_cnn_scratch.py

Goal:
Train a simple Convolutional Neural Network (CNN) from scratch for fish image classification.

Steps:
1. Import data generators from data_preprocessing.py
2. Build a CNN architecture manually
3. Compile and train the model
4. Save the best performing model automatically
5. Plot and save accuracy/loss graphs for analysis
"""

# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# Import data generators from package
from src.data_preprocessing import train_generator, val_generator

# =========================================================
# 2. CONFIGURATION
# =========================================================
IMG_SIZE = (224, 224)
EPOCHS = 15
LEARNING_RATE = 0.001

MODEL_DIR = "models"
RESULTS_DIR = "results/training_history"
LOGS_DIR = "logs"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# =========================================================
# 3. BUILD CNN MODEL FROM SCRATCH
# =========================================================
def build_cnn(input_shape, num_classes):
    """A simple CNN architecture for image classification."""
    model = Sequential([
        # --- 1st Convolution Block ---
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # --- 2nd Convolution Block ---
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # --- 3rd Convolution Block ---
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # --- Flatten + Dense Layers ---
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model

# =========================================================
# 4. COMPILE THE MODEL
# =========================================================
num_classes = len(train_generator.class_indices)
input_shape = IMG_SIZE + (3,)

cnn_model = build_cnn(input_shape, num_classes)

cnn_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n CNN Model Summary:\n")
cnn_model.summary()

# =========================================================
# 5. DEFINE CALLBACKS
# =========================================================
checkpoint_path = os.path.join(MODEL_DIR, "cnn_scratch_best.h5")

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# =========================================================
# 6. TRAIN THE MODEL
# =========================================================
print("\n Starting CNN Training...\n")

history = cnn_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

print("\n Training Complete! Best model saved at:", checkpoint_path)

# =========================================================
# 7. PLOT TRAINING HISTORY
# =========================================================
def plot_training_history(history, save_path):
    """Plot accuracy and loss curves and save them."""
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # --- Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training vs Validation Accuracy')

    # --- Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training vs Validation Loss')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
plot_path = os.path.join(RESULTS_DIR, f"cnn_history_{timestamp}.png")
plot_training_history(history, plot_path)
print(f"\n Training history saved at: {plot_path}")

# =========================================================
# 8. SAVE FINAL MODEL
# =========================================================
final_model_path = os.path.join(MODEL_DIR, "cnn_scratch_final.h5")
cnn_model.save(final_model_path)
print(f"\n Final model saved at: {final_model_path}")

print("\n All Done! CNN model trained successfully.")
