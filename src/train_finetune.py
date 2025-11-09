"""
train_finetune.py
-----------------
Fine-tune pretrained models (VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0)
that were trained using `train_pretrained_models.py`.

Usage:
    python -m src.train_finetune <model_name>

Example:
    python -m src.train_finetune efficientnetb0
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---- PATH CONFIGURATION ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "training_history")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n========== PATH CONFIGURATION ==========")
print(f"BASE_DIR   : {BASE_DIR}")
print(f"ROOT_DIR   : {ROOT_DIR}")
print(f"DATA_DIR   : {DATA_DIR}")
print(f"MODEL_SAVE : {MODEL_SAVE_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print("========================================\n")

# ---- DATA GENERATORS ------------------------------------------------------
def get_data_generators(img_size=(224, 224), batch_size=32):
    """Create train/val/test generators with light augmentation for fine-tuning."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"), target_size=img_size,
        batch_size=batch_size, class_mode="categorical", shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"), target_size=img_size,
        batch_size=batch_size, class_mode="categorical", shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"), target_size=img_size,
        batch_size=batch_size, class_mode="categorical", shuffle=False
    )
    return train_gen, val_gen, test_gen

# ---- PLOT FUNCTION --------------------------------------------------------
def plot_training_history(history, model_name, prefix="finetune"):
    """Plot and save accuracy/loss curves for a training history."""
    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title(f"{model_name.upper()} {prefix} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title(f"{model_name.upper()} {prefix} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{model_name}_{prefix}_history_{ts}.png")
    plt.savefig(path)
    plt.close()
    print(f" Saved plot: {path}")

# ---- FINE-TUNE FUNCTION ---------------------------------------------------
def fine_tune_model(model_name, unfreeze_layers=20, epochs=5, batch_size=32):
    """
    Load the saved transfer-learning model, unfreeze last N layers, fine-tune,
    evaluate on test set, save fine-tuned model in Keras format, and save plots.
    """
    model_name = model_name.lower()
    model_path = os.path.join(MODEL_SAVE_DIR, f"model_{model_name}.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Base model not found: {model_path} — run train_pretrained_models first.")

    print(f"\n Fine-tuning {model_name.upper()} — loading: {model_path}")
    model = load_model(model_path)
    print(" Model loaded successfully.")

    # Unfreeze last `unfreeze_layers` layers of the full model
    total_layers = len(model.layers)
    n = min(unfreeze_layers, total_layers)
    for layer in model.layers[: total_layers - n]:
        layer.trainable = False
    for layer in model.layers[total_layers - n:]:
        layer.trainable = True
    print(f"  Unfroze last {n} layers out of {total_layers} total layers.")

    # Recompile with small learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

    # Data generators
    train_gen, val_gen, test_gen = get_data_generators(batch_size=batch_size)

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)

    # Train
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[early_stop, reduce_lr], verbose=1)

    # Evaluate
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"\n Fine-tuned {model_name.upper()} — Test accuracy: {acc:.4f}, loss: {loss:.4f}")

    # Save fine-tuned model in modern Keras format to avoid HDF5 pickling errors
    out_path = os.path.join(MODEL_SAVE_DIR, f"model_{model_name}_finetuned.keras")
    model.save(out_path)
    print(f"  Fine-tuned model saved: {out_path}")

    # Save training history arrays (numpy) and CSV
    np.save(os.path.join(RESULTS_DIR, f"{model_name}_finetune_history.npy"), history.history)
    csv_path = os.path.join(RESULTS_DIR, f"{model_name}_finetune_history.csv")
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        keys = ["epoch", "accuracy", "val_accuracy", "loss", "val_loss"]
        writer.writerow(keys)
        for i in range(len(history.history.get("loss", []))):
            writer.writerow([
                i + 1,
                history.history.get("accuracy", [None])[i],
                history.history.get("val_accuracy", [None])[i],
                history.history.get("loss", [None])[i],
                history.history.get("val_loss", [None])[i],
            ])
    print(f"  Saved history CSV: {csv_path}")

    # Plot and save figures
    plot_training_history(history, model_name, prefix="finetune")

# ---- ENTRY POINT ----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.train_finetune <model_name>")
        print("Available: vgg16, resnet50, mobilenetv2, inceptionv3, efficientnetb0")
        sys.exit(1)

    model_arg = sys.argv[1].lower()
    fine_tune_model(model_arg)
