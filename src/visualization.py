import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader # Henter funksjoner fra dataloader
import numpy as np
import torchvision.transforms as transforms
import cv2
import argparse  # For kommandolinjeargumenter

# Importer dine egendefinerte datasett- og modellklasser
from dataloader import segDataset  # Sørg for at denne filen er i samme katalog eller juster importstien
from model import UNet  # Sørg for at denne filen inneholder din UNet-klasse definisjon

# Enhetskonfigurasjon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Bruker enhet: {device}")

# Parse kommandolinjeargumenter
parser = argparse.ArgumentParser(description='Inferens Skript')
parser.add_argument('--data_root', type=str, default=r'PATH_TIL_DATASETT', help='Rotkatalog for ditt datasett')
parser.add_argument('--model_path', type=str, default=r'PATH_TIL_MODELLFIL', help='Sti til din lagrede modell')
parser.add_argument('--batch_size', type=int, default=4, help='Batch-størrelse for DataLoader')
parser.add_argument('--patch_size', type=int, default=512, help='Flis-størrelse')
parser.add_argument('--threshold', type=float, default=0.5, help='Terskel for klassefordeling')
parser.add_argument('--mode', type=str, default='inference', choices=['train', 'inference'], help='Modus: train eller inference')
args = parser.parse_args()

# Parametere fra kommandolinjeargumenter
data_root = args.data_root
model_path = args.model_path
batch_size = args.batch_size
patch_size = args.patch_size
threshold = args.threshold
mode = args.mode  # 'train' eller 'inference'

# Transformasjoner (hvis noen)
transform = transforms.Compose([
    # Legg til nødvendige transformasjoner her
])

# Last inn testdatasettet
testing_root = os.path.join(data_root, 'testing')
print(f"Testdata rotkatalog: {testing_root}")

# Instansier datasettet med riktig rot og modus
testing_dataset = segDataset(root=testing_root, patch_size=patch_size, mode=mode, transform=transform)

# Egendefinert collate-funksjon for å håndtere posisjoner
def custom_collate_fn(batch):
    if len(batch) == 0:
        return None

    images, positions, image_indices = [], [], []
    if isinstance(batch[0], tuple) and len(batch[0]) == 4:  # Treningsmodus
        annotations = []
        for item in batch:
            images.append(item[0])
            annotations.append(item[1])
            positions.append(item[2])
            image_indices.append(item[3])
        images = torch.stack(images, 0)
        annotations = torch.stack(annotations, 0)
        return images, annotations, positions, image_indices
    else:  # Inferensmodus
        for item in batch:
            images.append(item[0])
            positions.append(item[1])
            image_indices.append(item[2])
        images = torch.stack(images, 0)
        return images, positions, image_indices

testing_dataloader = DataLoader(
    testing_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate_fn
)

# Initialiser modellen
model = UNet(n_channels=3, n_classes=2, bilinear=True).to(device)  # Juster bilinear etter behov
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Kjør inferens og behandle hvert bilde
with torch.no_grad():
    # Vi vil lagre prediksjoner for hvert bilde separat
    predictions_per_image = {}
    for batch in testing_dataloader:
        if batch is None:
            continue
        if mode == "train":
            inputs, _, positions, image_indices = batch
        else:  # Inferensmodus
            inputs, positions, image_indices = batch

        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)  # Beregn softmax sannsynligheter
        class1_probs = probs[:, 1, :, :]       # Hent sannsynligheter for klasse 1

        # Anvend terskel for å få binære prediksjoner
        preds = (class1_probs > threshold).long()
        preds = preds.cpu().numpy()

        batch_size_actual = inputs.size(0)
        for i in range(batch_size_actual):
            pred_patch = preds[i]
            x, y = positions[i]
            img_idx = image_indices[i]

            # Initialiser den rekonstruerte masken for dette bildet hvis ikke allerede gjort
            if img_idx not in predictions_per_image:
                height, width = testing_dataset.image_shapes[img_idx]
                predictions_per_image[img_idx] = np.zeros((height, width), dtype=np.uint8)

            # Plasser flisprediksjonen inn i den rekonstruerte masken
            predictions_per_image[img_idx][x:x+patch_size, y:y+patch_size] = pred_patch

# Nå overlay og lagre hvert bilde
for img_idx, reconstructed_mask in predictions_per_image.items():
    # Last inn det originale bildet
    original_image_path = testing_dataset.IMG_NAMES[img_idx]
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Sørg for at det originale bildet er samme størrelse som masken
    height, width = testing_dataset.image_shapes[img_idx]
    original_image = cv2.resize(original_image, (width, height))

    # Lag en fargemask hvor prediksjonene er 1
    color_mask = np.zeros_like(original_image)
    color_mask[reconstructed_mask == 1] = [0, 255, 0]  # Grønn farge

    # Kombiner det originale bildet og fargemasken med transparens
    alpha = 0.5  # Transparensfaktor
    overlayed_image = original_image.copy()
    mask_indices = reconstructed_mask == 1
    overlayed_image[mask_indices] = cv2.addWeighted(
        color_mask[mask_indices],
        alpha,
        original_image[mask_indices],
        1 - alpha,
        0
    )

    # Lagre det overlagrede bildet
    output_filename = os.path.basename(original_image_path)
    output_path = os.path.join('OVERLAYED_IMAGES_DIR', output_filename)
    os.makedirs('OVERLAYED_IMAGES_DIR', exist_ok=True)
    overlayed_image_bgr = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlayed_image_bgr)
    print(f"Overlagret bilde lagret til {output_path}")
