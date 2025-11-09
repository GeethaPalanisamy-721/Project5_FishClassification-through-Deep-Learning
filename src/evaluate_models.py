"""
evaluate_models.py
------------------
Evaluates all fine-tuned pretrained models on the test dataset.
Generates confusion matrices, classification metrics (accuracy, precision,
recall, F1-score), and summary CSV + bar chart visualizations.

Usage:
    python -m src.evaluate_models
"""

# ======== IMPORTS ==========================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======== PATH CONFIGURATION ==============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")

os.makedirs(CM_DIR, exist_ok=True)

print("\n========== PATH CONFIGURATION ==========")
print(f"DATA_DIR   : {DATA_DIR}")
print(f"MODEL_DIR  : {MODEL_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print("========================================\n")

# ======== LOAD TEST DATA ===================================================
def get_test_generator(img_size=(224, 224), batch_size=32):
    """
    Create a generator for test images.
    Only rescaling is done (no augmentation).
    """
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False  # Important: we want predictions in same order as filenames
    )
    return test_gen

# ======== EVALUATION FUNCTION =============================================
def evaluate_model(model_path, test_gen):
    """
    Loads a model, evaluates it on the test set,
    returns metrics and saves confusion matrix plot.
    """
    model_name = os.path.basename(model_path).replace("model_", "").replace("_finetuned.keras", "")
    print(f"\n Evaluating model: {model_name.upper()}")

    # --- Load model from disk ---
    model = load_model(model_path)
    print("  Model loaded successfully.")

    # --- Get predictions ---
    # predict() gives probabilities for each class
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes  # ground truth labels (from directory structure)
    class_labels = list(test_gen.class_indices.keys())

    # --- Compute metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f" Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = os.path.join(CM_DIR, f"{model_name}_cm.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f" Saved confusion matrix: {cm_path}")

    # --- Save classification report (optional, readable table) ---
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    # Save inside results/classification_report folder
    CLASS_REPORT_DIR = os.path.join(RESULTS_DIR, "classification_report")
    os.makedirs(CLASS_REPORT_DIR, exist_ok=True)

    report_path = os.path.join(CLASS_REPORT_DIR, f"{model_name}_classification_report.csv")
    report_df.to_csv(report_path)
    print(f" Saved detailed report: {report_path}")

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

# ======== MAIN DRIVER =====================================================
def evaluate_all_models():
    """
    Evaluate all *_finetuned.keras models present in models/ folder.
    Creates summary CSV and comparison plots.
    """
    test_gen = get_test_generator()
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_finetuned.keras")]

    if not model_files:
        print(" No fine-tuned models found in 'models/'. Run train_finetune.py first.")
        return

    print(f"Found {len(model_files)} fine-tuned models: {model_files}")

    results = []
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        metrics = evaluate_model(model_path, test_gen)
        results.append(metrics)

    # --- Save metrics summary to CSV ---
    df = pd.DataFrame(results)

    EVAL_SUMMARY_DIR = os.path.join(RESULTS_DIR, "evaluation_summary")
    os.makedirs(EVAL_SUMMARY_DIR, exist_ok=True)

    summary_path = os.path.join(EVAL_SUMMARY_DIR, "metrics_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n Saved metrics summary: {summary_path}")


    # --- Create comparison bar plots ---
    plt.figure(figsize=(10, 5))
    plt.bar(df["Model"], df["Accuracy"])
    plt.title("Model Comparison - Accuracy")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_accuracy_comparison.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(df["Model"], df["F1"])
    plt.title("Model Comparison - F1 Score")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_f1_comparison.png"))
    plt.close()

    print(" Saved accuracy and F1 comparison plots.")

# ======== ENTRY POINT =====================================================
if __name__ == "__main__":
    evaluate_all_models()
