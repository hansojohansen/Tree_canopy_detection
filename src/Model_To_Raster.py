import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import rasterio
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Custom imports
from dataloader import segDataset
from model import UNet

B_value = True
Tile_Size = 124
Batch1_Size= 6

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Argument parser
parser = argparse.ArgumentParser(description='Inference Script')
parser.add_argument('--data_root', type=str, default=r'C:\Users\milge\OneDrive\Dokumenter\Johansen\Bachelor Oppgave\Database', help='Root directory of your dataset')
parser.add_argument('--model_path', type=str, default=r'C:\Users\milge\OneDrive\Dokumenter\Johansen\Bachelor Oppgave\saved_models\training_20241108_095403_focalloss_bilinear_124_B64\best_model.pth', help='Path to your saved model')
parser.add_argument('--batch_size', type=int, default=Batch1_Size, help='Batch size for DataLoader')
parser.add_argument('--patch_size', type=int, default=Tile_Size, help='Patch size')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification')
args = parser.parse_args()

# Parameters
data_root = args.data_root
model_path = args.model_path
batch_size = args.batch_size
patch_size = args.patch_size
threshold = args.threshold

# Output directory
model_name = os.path.basename(os.path.dirname(model_path))
output_dir = os.path.join("Predicted_Mask_Dir_Individuell", model_name)
os.makedirs(output_dir, exist_ok=True)

# Transformations
transform = transforms.Compose([])

# Load dataset
testing_root = os.path.join(data_root, 'testing')
print(f"Test data root: {testing_root}")
testing_dataset = segDataset(root=testing_root, patch_size=patch_size, mode='train', transform=transform)

# Custom collate function
def custom_collate_fn(batch):
    if len(batch) == 0:
        return None

    images, annotations, positions, image_indices = [], [], [], []
    for item in batch:
        images.append(item[0])
        annotations.append(item[1])
        positions.append(item[2])
        image_indices.append(item[3])
    images = torch.stack(images, 0)
    annotations = torch.stack(annotations, 0)
    return images, annotations, positions, image_indices

# DataLoader
testing_dataloader = DataLoader(
    testing_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate_fn
)

# Initialize model
model = UNet(n_channels=3, n_classes=2, bilinear=B_value).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Initialize confusion matrices per image
cm_per_image = {}  # Dictionary to store confusion matrices for each image
predictions_per_image = {}

# Run inference
with torch.no_grad():
    for batch in testing_dataloader:
        if batch is None:
            continue

        inputs, annotations, positions, image_indices = batch
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        class1_probs = probs[:, 1, :, :]

        preds = (class1_probs > threshold).long()
        preds_np = preds.cpu().numpy()
        annotations_np = annotations.cpu().numpy()

        batch_size_actual = inputs.size(0)
        for i in range(batch_size_actual):
            pred_patch = preds_np[i]
            true_patch = annotations_np[i]

            x, y = positions[i]
            img_idx = image_indices[i]
            img_path = testing_dataset.IMG_NAMES[img_idx]

            # Extract image name without extension to use as identifier
            image_name = os.path.splitext(os.path.basename(img_path))[0]

            # Initialize confusion matrix for the image if not present
            if image_name not in cm_per_image:
                cm_per_image[image_name] = np.zeros((2, 2), dtype=np.int64)

            # Update confusion matrix for the current image
            cm_batch = confusion_matrix(true_patch.flatten(), pred_patch.flatten(), labels=[0, 1])
            cm_per_image[image_name] += cm_batch

            if img_idx not in predictions_per_image:
                height, width = testing_dataset.image_shapes[img_idx]
                predictions_per_image[img_idx] = np.zeros((height, width), dtype=np.uint8)

            predictions_per_image[img_idx][x:x+patch_size, y:y+patch_size] = pred_patch

# Output directory for individual images
for image_name, cm_total in cm_per_image.items():
    # Compute overall metrics from the confusion matrix
    accuracy = (cm_total[0, 0] + cm_total[1, 1]) / cm_total.sum()
    precision = cm_total[1, 1] / (cm_total[0, 1] + cm_total[1, 1] + 1e-8)
    recall = cm_total[1, 1] / (cm_total[1, 0] + cm_total[1, 1] + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Convert metrics to percentages
    accuracy_pct = accuracy * 100
    precision_pct = precision * 100
    recall_pct = recall * 100
    f1_pct = f1 * 100

    # Normalize the confusion matrix by rows (actual classes)
    cm_normalized = cm_total.astype('float') / cm_total.sum(axis=1)[:, np.newaxis]

    # Create a subdirectory for the image
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Save confusion matrix as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["Predikert Annet", "Predikert Tredekke"],
                yticklabels=["Faktisk Annet", "Faktisk Tredekke"])
    plt.title(f"Forvirringsmatrise for - {image_name}")
    plt.ylabel('Annotering')
    plt.xlabel('Modellens prediksjon')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(image_output_dir, f"confusion_matrix_normalized_{image_name}.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Save metrics as text file
    metrics_path = os.path.join(image_output_dir, f"metrics_{image_name}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy_pct:.2f}%\n")
        f.write(f"Precision: {precision_pct:.2f}%\n")
        f.write(f"Recall: {recall_pct:.2f}%\n")
        f.write(f"F1 Score: {f1_pct:.2f}%\n")
        f.write("\nConfusion Matrix (Counts):\n")
        f.write(np.array2string(cm_total, separator=", "))
        f.write("\n\nConfusion Matrix (Normalized by Actual Classes):\n")
        f.write(np.array2string(cm_normalized, formatter={'float_kind':lambda x: "%.4f" % x}))

    print(f"Metrics and confusion matrix saved for image: {image_name}")

# Save masks as georeferenced TIFF files
for img_idx, reconstructed_mask in predictions_per_image.items():
    original_image_path = testing_dataset.IMG_NAMES[img_idx]
    with rasterio.open(original_image_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs

    meta.update({
        'count': 1,
        'dtype': 'uint8',
        'compress': 'lzw'
    })

    height, width = testing_dataset.image_shapes[img_idx]
    reconstructed_mask = reconstructed_mask[:height, :width]

    # Extract image name
    image_name = os.path.splitext(os.path.basename(original_image_path))[0]
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    output_filename = f"{image_name}_mask.tif"
    output_path = os.path.join(image_output_dir, output_filename)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(reconstructed_mask.astype(rasterio.uint8), 1)
    print(f"Predicted mask saved to {output_path}")
