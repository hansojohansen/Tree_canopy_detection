import os
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the directories containing the ground truth and predicted rasters
file_loc_annotation = r'C:\Users\milge\OneDrive\Dokumenter\Johansen\Bachelor Oppgave\Database\testing\annotations'
file_loc_test_raster = r'C:\Users\milge\OneDrive\Dokumenter\Johansen\Bachelor Oppgave\AR5'

# Output directory
output_dir = 'Metrics_Output'
os.makedirs(output_dir, exist_ok=True)

# Function to process rasters in blocks
def process_rasters_in_blocks(annotation_path, prediction_path, block_size=512):
    with rasterio.open(annotation_path) as src_true, rasterio.open(prediction_path) as src_pred:
        if src_true.width != src_pred.width or src_true.height != src_pred.height:
            raise ValueError(f"The rasters do not have the same dimensions: {annotation_path} vs {prediction_path}")

        # Get the dimensions
        width = src_true.width
        height = src_true.height

        # Initialize confusion matrix
        cm = np.zeros((2, 2), dtype=np.int64)

        # Iterate over blocks
        for y in range(0, height, block_size):
            block_height = min(block_size, height - y)
            for x in range(0, width, block_size):
                block_width = min(block_size, width - x)

                window = Window(x, y, block_width, block_height)

                # Read blocks
                true_data = src_true.read(1, window=window)
                pred_data = src_pred.read(1, window=window)

                # Handle no-data values
                no_data_value_true = src_true.nodata
                no_data_value_pred = src_pred.nodata

                mask = np.ones(true_data.shape, dtype=bool)
                if no_data_value_true is not None:
                    mask &= (true_data != no_data_value_true)
                if no_data_value_pred is not None:
                    mask &= (pred_data != no_data_value_pred)

                # If there are no valid data points in this block, skip
                if not np.any(mask):
                    continue

                true_block = true_data[mask].astype(np.uint8)
                pred_block = pred_data[mask].astype(np.uint8)

                # Compute confusion matrix for the block
                cm_block = confusion_matrix(true_block.flatten(), pred_block.flatten(), labels=[0, 1])
                cm += cm_block

        return cm

# Get list of files in the annotation and prediction directories
annotation_files = [f for f in os.listdir(file_loc_annotation) if f.endswith('.tif')]
prediction_files = [f for f in os.listdir(file_loc_test_raster) if f.endswith('.tif')]

# Find matching files
matching_files = set(annotation_files).intersection(prediction_files)

if not matching_files:
    raise ValueError("No matching files found between annotation and prediction directories.")

# Initialize combined confusion matrix
combined_cm = np.zeros((2, 2), dtype=np.int64)

# Loop over matching files
for filename in matching_files:
    annotation_path = os.path.join(file_loc_annotation, filename)
    prediction_path = os.path.join(file_loc_test_raster, filename)

    # Process rasters in blocks
    cm = process_rasters_in_blocks(annotation_path, prediction_path, block_size=512)

    # Add to combined confusion matrix
    combined_cm += cm

    # Calculate metrics
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    total = cm.sum()
    accuracy = (TP + TN) / total if total != 0 else 0
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Convert metrics to percentages
    accuracy_pct = accuracy * 100
    precision_pct = precision * 100
    recall_pct = recall * 100
    f1_pct = f1 * 100

    # Normalize the confusion matrix
    with np.errstate(all='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # Create output directory for this file
    file_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
    os.makedirs(file_output_dir, exist_ok=True)

    # Save confusion matrix as an image
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["Annet", "Skog"],
                yticklabels=["Faktisk Annet", "Faktisk Tredekke"])
    plt.title(f"Forvirringsmatrise for - {filename}")
    plt.ylabel('Annotering')
    plt.xlabel('AR5')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(file_output_dir, "confusion_matrix_normalized.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Save metrics as a text file
    metrics_path = os.path.join(file_output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy_pct:.2f}%\n")
        f.write(f"Precision: {precision_pct:.2f}%\n")
        f.write(f"Recall: {recall_pct:.2f}%\n")
        f.write(f"F1 Score: {f1_pct:.2f}%\n")
        f.write("\nConfusion Matrix (Counts):\n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n\nConfusion Matrix (Normalized by Actual Classes):\n")
        f.write(np.array2string(cm_normalized, formatter={'float_kind':lambda x: "%.4f" % x}))

    print(f"Metrics and confusion matrix saved for file: {filename}")

# After processing all files, compute combined metrics
TP = combined_cm[1, 1]
TN = combined_cm[0, 0]
FP = combined_cm[0, 1]
FN = combined_cm[1, 0]

total = combined_cm.sum()
combined_accuracy = (TP + TN) / total if total != 0 else 0
combined_precision = TP / (TP + FP + 1e-8)
combined_recall = TP / (TP + FN + 1e-8)
combined_f1 = 2 * combined_precision * combined_recall / (combined_precision + combined_recall + 1e-8)

# Convert metrics to percentages
combined_accuracy_pct = combined_accuracy * 100
combined_precision_pct = combined_precision * 100
combined_recall_pct = combined_recall * 100
combined_f1_pct = combined_f1 * 100

# Normalize the combined confusion matrix
with np.errstate(all='ignore'):
    combined_cm_normalized = combined_cm.astype('float') / combined_cm.sum(axis=1)[:, np.newaxis]
combined_cm_normalized = np.nan_to_num(combined_cm_normalized)

# Save combined confusion matrix as an image
plt.figure(figsize=(6, 5))
sns.heatmap(combined_cm_normalized, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=["Annet", "Skog"],
            yticklabels=["Faktisk Annet", "Faktisk Tredekke"])
plt.title("Forvirringsmatrise")
plt.ylabel('Annotering')
plt.xlabel('AR5')
plt.tight_layout()
combined_confusion_matrix_path = os.path.join(output_dir, "combined_confusion_matrix_normalized.png")
plt.savefig(combined_confusion_matrix_path)
plt.close()

# Save combined metrics as a text file
combined_metrics_path = os.path.join(output_dir, "combined_metrics.txt")
with open(combined_metrics_path, "w") as f:
    f.write(f"Accuracy: {combined_accuracy_pct:.2f}%\n")
    f.write(f"Precision: {combined_precision_pct:.2f}%\n")
    f.write(f"Recall: {combined_recall_pct:.2f}%\n")
    f.write(f"F1 Score: {combined_f1_pct:.2f}%\n")
    f.write("\nCombined Confusion Matrix (Counts):\n")
    f.write(np.array2string(combined_cm, separator=", "))
    f.write("\n\nCombined Confusion Matrix (Normalized by Actual Classes):\n")
    f.write(np.array2string(combined_cm_normalized, formatter={'float_kind':lambda x: "%.4f" % x}))

print("Combined metrics and confusion matrix saved.")
