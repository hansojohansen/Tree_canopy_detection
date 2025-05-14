import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import rasterio
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataloader import segDataset
from model import UNet

# Enhetskonfigurasjon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Bruker enhet: {device}")

# Les kommandolinjeargumenter
parser = argparse.ArgumentParser(description='Inferensskript med forvirringsmatrise og maskelagring')
parser.add_argument('--data_root', type=str, default=r'C:\Users\milge\OneDrive\Dokumenter\Johansen\Bachelor Oppgave\Database', help='Rotkatalog for ditt datasett')
parser.add_argument('--model_path', type=str, default=r'C:\Users\milge\OneDrive\Dokumenter\Johansen\Bachelor Oppgave\saved_models\training_20241108_095403_focalloss_bilinear_124_B64\best_model.pth', help='Sti til din lagrede modell')
parser.add_argument('--batch_size', type=int, default=4, help='Batch-størrelse for DataLoader')
parser.add_argument('--patch_size', type=int, default=124, help='Flisstørrelse')
parser.add_argument('--threshold', type=float, default=0.5, help='Terskel for klasseseparasjon')
parser.add_argument('--save_dir', type=str, default='PREDIKTERTE_MASKER_DIR', help='Katalog for å lagre predikerte masker')
args = parser.parse_args()

# Parametere fra kommandolinjeargumenter
data_root = args.data_root
model_path = args.model_path
batch_size = args.batch_size
patch_size = args.patch_size
threshold = args.threshold
save_dir = args.save_dir

# Bruk modellenavn for å lage en unik lagringskatalog
model_name = os.path.basename(os.path.dirname(model_path))  # Extract the folder name for the model
output_dir = os.path.join(save_dir, model_name)
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Transformasjoner
transform = transforms.Compose([])

# Last inn testdatasettet
testing_root = os.path.join(data_root, 'testing')
print(f"Testdata rotkatalog: {testing_root}")

testing_dataset = segDataset(root=testing_root, patch_size=patch_size, mode='train', transform=transform)

# Egendefinert collate-funksjon
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

# Opprett DataLoader
testing_dataloader = DataLoader(
    testing_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate_fn
)

# Initialiser modellen
model = UNet(n_channels=3, n_classes=2, bilinear=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Initialiser lister for å samle etiketter
true_labels_file = os.path.join(output_dir, 'true_labels.npy')
predicted_labels_file = os.path.join(output_dir, 'predicted_labels.npy')

if os.path.exists(true_labels_file):
    os.remove(true_labels_file)
if os.path.exists(predicted_labels_file):
    os.remove(predicted_labels_file)

# Kjør inferens
with torch.no_grad():
    predictions_per_image = {}
    for batch in testing_dataloader:
        if batch is None:
            continue

        inputs, annotations, positions, image_indices = batch

        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        class1_probs = probs[:, 1, :, :]

        preds = (class1_probs > threshold).long()
        preds = preds.cpu().numpy()

        batch_size_actual = inputs.size(0)
        for i in range(batch_size_actual):
            pred_patch = preds[i]
            true_patch = annotations[i].cpu().numpy()
            x, y = positions[i]
            img_idx = image_indices[i]

            # Append true and predicted labels incrementally
            with open(true_labels_file, 'ab') as f_true, open(predicted_labels_file, 'ab') as f_pred:
                np.save(f_true, true_patch.flatten())
                np.save(f_pred, pred_patch.flatten())

            # Rekonstruer fullstendige bildemasker
            if img_idx not in predictions_per_image:
                height, width = testing_dataset.image_shapes[img_idx]
                predictions_per_image[img_idx] = np.zeros((height, width), dtype=np.uint8)

            predictions_per_image[img_idx][x:x+patch_size, y:y+patch_size] = pred_patch

        # Clear memory
        del inputs, annotations, outputs, probs, preds
        torch.cuda.empty_cache()

# Les lagrede etiketter og beregn forvirringsmatrise
all_true_labels = np.load(true_labels_file, allow_pickle=True)
all_predicted_labels = np.load(predicted_labels_file, allow_pickle=True)

cm = confusion_matrix(np.concatenate(all_true_labels), np.concatenate(all_predicted_labels))
cm_normalized = confusion_matrix(np.concatenate(all_true_labels), np.concatenate(all_predicted_labels), normalize='true')

# Definer klassenavn
class_names = ['Annet', 'TreDekke']

# Plot forvirringsmatriser
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Forvirringsmatrise (Testsett)')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_test_set.png'))
plt.close()

disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
disp_normalized.plot(cmap=plt.cm.Blues)
plt.title('Normalisert forvirringsmatrise (Testsett)')
plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix_test_set.png'))
plt.close()

print("Forvirringsmatrise er beregnet og lagret.")

# Lagre rekonstruerte masker
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

    output_filename = os.path.basename(original_image_path)
    output_filename = os.path.splitext(output_filename)[0] + '_mask.tif'
    output_path = os.path.join(output_dir, output_filename)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(reconstructed_mask.astype(rasterio.uint8), 1)
    print(f"Predikert maske lagret til {output_path}")
