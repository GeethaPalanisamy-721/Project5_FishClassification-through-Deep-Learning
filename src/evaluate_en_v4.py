"""
evaluate_en_v4.py

Generates a complete evaluation summary (cluster report) for the final fine-tuned model.
Includes classification metrics, confusion matrix visualizations, and exports results as CSV.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from src.data_preprocessing import create_data_generators

# =========================================================
# CONFIGURATION
# =========================================================
MODEL_NAME = "efficientnetb0_finetuned_v4_final.keras"
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", MODEL_NAME)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

# Subdirectories for better organization
CLASSIFICATION_REPORT_DIR = os.path.join(RESULTS_DIR, "classification_reports")
CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "evaluation_summary")

# Create all directories if they don’t exist
os.makedirs(CLASSIFICATION_REPORT_DIR, exist_ok=True)
os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# =========================================================
# 1️. Load test data generator
# =========================================================
print("Loading test data generator...")
_, _, test_generator = create_data_generators()

# =========================================================
# 2️. Load fine-tuned model
# =========================================================
print(f" Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# =========================================================
# 3️. Predict on test set
# =========================================================
print("\n Running predictions on test data...")
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Predictions (probabilities)
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# =========================================================
# 4️. Compute key metrics (overall)
# =========================================================
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

print("\n========== FINAL MODEL SUMMARY ==========")
print(f" Model Name   : {MODEL_NAME}")
print(f" Test Accuracy: {acc:.4f}")
print(f" Precision    : {precision:.4f}")
print(f" Recall       : {recall:.4f}")
print(f" F1 Score     : {f1:.4f}")
print("=========================================")

# =========================================================
# 5️. Detailed classification report (per class)
# =========================================================
report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Save as CSV inside classification_reports folder
csv_path = os.path.join(CLASSIFICATION_REPORT_DIR, f"classification_report_{MODEL_NAME.replace('.keras','')}.csv")
report_df.to_csv(csv_path, index=True)
print(f"\n Classification report saved to: {csv_path}")

# =========================================================
# 6️. Confusion matrix
# =========================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.title(f"Confusion Matrix: {MODEL_NAME}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save confusion matrix inside confusion_matrices folder
cm_path = os.path.join(CONFUSION_MATRIX_DIR, f"confusion_matrix_{MODEL_NAME.replace('.keras','')}.png")
plt.savefig(cm_path)
plt.show()

print(f" Confusion matrix saved to: {cm_path}")

# =========================================================
# 7️. Save summary metrics as CSV (for report)
# =========================================================
summary_data = {
    "Model": [MODEL_NAME],
    "Accuracy": [acc],
    "Precision (macro)": [precision],
    "Recall (macro)": [recall],
    "F1-score (macro)": [f1]
}
summary_df = pd.DataFrame(summary_data)

# Save inside evaluation_summary folder
summary_csv = os.path.join(SUMMARY_DIR, "final_model_summary.csv")
summary_df.to_csv(summary_csv, index=False)

print(f" Overall summary saved to: {summary_csv}")
print("\n Evaluation complete! Use these files in your final report.")
