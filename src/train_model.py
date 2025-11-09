"""
train_model.py
----------------
This script performs the **base model training** for fish image classification.

It:
1. Loads and augments training, validation, and test datasets.
2. Builds a CNN model (EfficientNet, ResNet, VGG, etc.).
3. Trains it with callbacks like EarlyStopping and ReduceLROnPlateau.
4. Saves the best model in `.keras` format for later fine-tuning.
5. Saves plots and logs for training metrics.

To run:
    python -m src.train_model efficientnetb0
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import sys
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import (
    EfficientNetB0, ResNet50, MobileNetV2, VGG16, InceptionV3
)
import matplotlib.pyplot as plt
from datetime import datetime

# ================================================================
# PATH CONFIGURATION
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))  # project root
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "training_history")

# Ensure folders exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n========== PATH CONFIGURATION ==========")
print(f"BASE_DIR   : {BASE_DIR}")
print(f"ROOT_DIR   : {ROOT_DIR}")
print(f"DATA_DIR   : {DATA_DIR}")
print(f"MODEL_SAVE : {MODEL_SAVE_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print("========================================\n")

# ================================================================
# DATA GENERATORS
# ================================================================
def get_data_generators(img_size=(224, 224), batch_size=32):
    """
    Creates ImageDataGenerators for training, validation, and test sets.
    Includes augmentation for training and rescaling for all.
    """
    datagen_train = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    datagen_val = ImageDataGenerator(rescale=1.0 / 255)
    datagen_test = ImageDataGenerator(rescale=1.0 / 255)

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    train_gen = datagen_train.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    val_gen = datagen_val.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    test_gen = datagen_test.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )

    return train_gen, val_gen, test_gen

# ================================================================
# MODEL BUILDER
# ================================================================
def build_model(model_choice, input_shape=(224, 224, 3), num_classes=11):
    """
    Builds a CNN model using a selected pretrained base (transfer learning).
    All models use ImageNet weights and add a custom classification head.
    """
    model_choice = model_choice.lower()

    # --- Choose the backbone ---
    if model_choice == "efficientnetb0":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_choice == "resnet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_choice == "mobilenetv2":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_choice == "vgg16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_choice == "inceptionv3":
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f" Unsupported model choice: {model_choice}")

    # --- Custom classification head ---
    x = base_model.output
    x = GlobalAveragePooling2D()(x)       # convert feature maps → 1D vector
    x = Dropout(0.4)(x)                   # prevent overfitting
    predictions = Dense(num_classes, activation="softmax")(x)  # final layer

    model = Model(inputs=base_model.input, outputs=predictions)

    # --- Compile model ---
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ================================================================
# PLOT TRAINING HISTORY
# ================================================================
def plot_training_history(history, model_choice):
    """Saves training and validation accuracy/loss plots."""
    plt.figure(figsize=(12, 5))

    # --- Accuracy plot ---
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # --- Loss plot ---
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plot_path = os.path.join(RESULTS_DIR, f"{model_choice}_training_plot_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f" Training plots saved at: {plot_path}")

# ================================================================
# TRAINING FUNCTION
# ================================================================
def run_training(model_choice, epochs=10):
    """
    Main training function:
    - Loads data generators
    - Builds selected model
    - Trains with callbacks
    - Saves results & model
    """
    train_gen, val_gen, test_gen = get_data_generators()
    num_classes = len(train_gen.class_indices)

    model = build_model(model_choice, num_classes=num_classes)

    # Save model in .keras format (for fine-tuning compatibility)
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"model_{model_choice}.keras")

    # --- Callbacks ---
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    print(f"\n Starting training for {model_choice}...\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )

    print(f"\n Training complete for {model_choice}!")
    print(f" Best model saved to: {checkpoint_path}\n")

    # --- Save history (numpy + CSV) ---
    np.save(os.path.join(RESULTS_DIR, f"{model_choice}_history.npy"), history.history)

    csv_path = os.path.join(RESULTS_DIR, f"{model_choice}_history.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "val_accuracy", "loss", "val_loss"])
        for i in range(len(history.history["accuracy"])):
            writer.writerow([
                i + 1,
                history.history["accuracy"][i],
                history.history["val_accuracy"][i],
                history.history["loss"][i],
                history.history["val_loss"][i],
            ])
    print(f" Metrics saved to CSV: {csv_path}")

    # --- Plot accuracy/loss curves ---
    plot_training_history(history, model_choice)

    # --- Final save (redundant but safe for finetune) ---
    keras_save_path = os.path.join(MODEL_SAVE_DIR, f"model_{model_choice}.keras")
    model.save(keras_save_path)
    print(f" Model saved successfully as: {keras_save_path}")

# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.train_model <model_choice>")
        print("Available models: vgg16, resnet50, mobilenetv2, inceptionv3, efficientnetb0")
        sys.exit(1)

    model_choice = sys.argv[1]
    run_training(model_choice)
