"""
train_finetune_v4.py
--------------------
This script fine-tunes pretrained CNN models (e.g., EfficientNetB0, ResNet50, etc.)
for fish species classification.

It assumes you have:
- src/data_preprocessing.py with get_data_generators()
- src/train_model.py with build_model()

"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ===============================================
# Import helper functions from your project
# ===============================================
# get_data_generators → loads data for training, validation, and testing
# build_model → builds a pretrained CNN (VGG16, ResNet50, EfficientNetB0, etc.)
from src.data_preprocessing import get_data_generators
from src.train_model import build_model

# ===============================================
# CONFIGURATION
# ===============================================
EPOCHS = 10            # Fine-tuning epochs
UNFREEZE_LAYERS = 20   # How many last layers to unfreeze
LEARNING_RATE = 1e-5   # Lower LR for fine-tuning
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "training_history")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================================
# FINE-TUNING FUNCTION
# ===============================================
def fine_tune_model(model_name: str):
    """
    Loads a pretrained model (already trained earlier), unfreezes last few layers,
    and fine-tunes on the dataset for improved accuracy.
    """

    # -------------------------------
    # STEP 1: Load dataset
    # -------------------------------
    print("\n Loading dataset using get_data_generators()...")
    train_gen, val_gen, test_gen = get_data_generators()
    num_classes = len(train_gen.class_indices)
    print(f"Classes found: {list(train_gen.class_indices.keys())}\n")

    # -------------------------------
    # STEP 2: Load pretrained model
    # -------------------------------
    print(f" Loading pretrained {model_name.upper()} model...")
    base_model_path = os.path.join(MODEL_DIR, f"model_{model_name}.keras")

    if not os.path.exists(base_model_path):
        print(f" Error: Base model not found at {base_model_path}")
        print("Please train the base model first using: python -m src.train_model", model_name)
        sys.exit(1)

    # Load the trained model
    model = tf.keras.models.load_model(base_model_path)
    print(" Model loaded successfully.")

    # -------------------------------
    # STEP 3: Unfreeze last few layers
    # -------------------------------
    for layer in model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    print(f"Unfroze last {UNFREEZE_LAYERS} layers for fine-tuning.")

    # -------------------------------
    # STEP 4: Compile model
    # -------------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(" Model compiled for fine-tuning.\n")

    # -------------------------------
    # STEP 5: Set up callbacks
    # -------------------------------
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, f"{model_name}_finetuned_v4.keras"),
                        save_best_only=True, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
    ]

    # -------------------------------
    # STEP 6: Train (fine-tune) model
    # -------------------------------
    print(f" Starting fine-tuning for {model_name.upper()}...\n")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # -------------------------------
    # STEP 7: Evaluate on test data
    # -------------------------------
    print("\n Evaluating fine-tuned model on test data...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f" Final Test Accuracy: {test_acc:.4f}")

    # -------------------------------
    # STEP 8: Save the final model
    # -------------------------------
    fine_tuned_path = os.path.join(MODEL_DIR, f"{model_name}_finetuned_v4_final.keras")
    model.save(fine_tuned_path)
    print(f" Fine-tuned model saved to: {fine_tuned_path}")

    # -------------------------------
    # STEP 9: Plot training history
    # -------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name.upper()} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name.upper()} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    results_path = os.path.join(RESULTS_DIR, f"{model_name}_finetune_history_v4.png")
    plt.savefig(results_path)
    plt.show()

    print(f" Training history plot saved at: {results_path}")


# ===============================================
# MAIN EXECUTION (CLI)
# ===============================================
if __name__ == "__main__":
    # Example usage:
    # python -m src.train_finetune_v4 efficientnetb0
    if len(sys.argv) != 2:
        print("Usage: python -m src.train_finetune_v4 <model_name>")
        print("Example: python -m src.train_finetune_v4 efficientnetb0")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    fine_tune_model(model_name)
