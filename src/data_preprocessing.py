"""
data_preprocessing.py

This module handles:
1. Dataset summary and sanity checks.
2. Loading train, validation, and test datasets using ImageDataGenerator.
3. Applying rescaling and augmentation to enhance model robustness.
4. Visualizing augmented images (optional for local environments).
5. Exporting generators for reuse in training scripts.
"""

import os
import logging
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================================================
# CONFIGURATION
# =========================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")  # relative to project root

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# =========================================================
# LOGGING SETUP
# =========================================================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# =========================================================
# 1. Dataset Summary Function
# =========================================================
def summarize_data_folders(base_path: str):
    """Prints number of folders (classes) and total image counts for train/val/test."""
    logger.info(f"Checking dataset structure in: {os.path.abspath(base_path)}")
    logger.info("=" * 60)

    for subset in ["train", "val", "test"]:
        subset_path = os.path.join(base_path, subset)
        if not os.path.exists(subset_path):
            logger.warning(f"Missing folder: {subset_path}")
            continue

        class_folders = [
            f for f in os.listdir(subset_path)
            if os.path.isdir(os.path.join(subset_path, f))
        ]
        num_classes = len(class_folders)
        total_images = sum(
            len(files) for _, _, files in os.walk(subset_path)
        )

        logger.info(f"{subset.upper():<6} ➜ Classes: {num_classes}, Images: {total_images}")

# =========================================================
# 2. Create Image Data Generators
# =========================================================
def create_data_generators():
    """Creates and returns train, validation, and test data generators."""
    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.15,
        fill_mode="nearest"
    )

    # Only rescaling for validation and test sets
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Flow from directory
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False
    )

    return train_generator, val_generator, test_generator

# =========================================================
# 3. Visualization Function
# =========================================================
def visualize_augmentations(generator, num_images=5):
    """
    Displays a few augmented samples from the training generator.
    """
    try:
        x_batch, y_batch = next(generator)
        plt.figure(figsize=(12, 6))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(x_batch[i])
            plt.axis("off")
            class_label = list(generator.class_indices.keys())[y_batch[i].argmax()]
            plt.title(f"Augmented\n{class_label}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.warning(f"Visualization skipped: {e}")

# =========================================================
# 4. Execute when run directly
# =========================================================
if __name__ == "__main__":
    logger.info("Running data preprocessing pipeline...")

    summarize_data_folders(DATA_DIR)
    train_generator, val_generator, test_generator = create_data_generators()

    visualize_augmentations(train_generator, num_images=5)

    logger.info("\n Data Preprocessing Complete!")
    logger.info(f"Classes found: {train_generator.class_indices}")
    logger.info(f"Image shape: {IMG_SIZE + (3,)}")
    logger.info(
        f"Train batches: {len(train_generator)}, "
        f"Val batches: {len(val_generator)}, "
        f"Test batches: {len(test_generator)}"
    )

    logger.info("\nTo use in training scripts, import this file as a module:")
    logger.info("from src.data_preprocessing import train_generator, val_generator, test_generator")

# =========================================================
# 5. Expose for imports
# =========================================================
train_generator, val_generator, test_generator = create_data_generators()

# =========================================================
# 6. New: Flexible data loader for fine-tuning models (train_finetune_v4)
# =========================================================
def get_data_generators(
    data_dir=None,
    img_size=(224, 224),
    batch_size=32,
    augment=True
):
    """
    A flexible data loading function designed for fine-tuning and transfer learning.
    It allows dynamic dataset paths and optional augmentation.
    
    Parameters
    ----------
    data_dir : str or None
        Base directory containing 'train', 'val', and 'test' folders.
        If None, it uses the default path (../data).
    img_size : tuple
        Target size of images (width, height).
    batch_size : int
        Number of samples per batch.
    augment : bool
        Whether to apply data augmentation on the training set.

    Returns
    -------
    tuple : (train_generator, val_generator, test_generator)
        The Keras ImageDataGenerators ready for training.
    """
    if data_dir is None:
        data_dir = DATA_DIR  # fallback to default project data path

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Augmentation for training
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            shear_range=0.15,
            fill_mode="nearest"
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # No augmentation for validation/test
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Flow from directory
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    print(f"\n Loaded data from: {data_dir}")
    print(f"Train: {train_generator.samples} | Val: {val_generator.samples} | Test: {test_generator.samples}")
    print(f"Classes: {list(train_generator.class_indices.keys())}\n")

    return train_generator, val_generator, test_generator
