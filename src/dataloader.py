import numpy as np
import torch
import torchvision.transforms as transforms
from glob import glob
import os
import rasterio
from rasterio.enums import Resampling
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import random

''' 
Velg størrelse på ruter, trenings- eller testmodus (inference), og andre parametere dersom nødvendig 
'''
class segDataset(torch.utils.data.Dataset):
    def __init__(self, root, patch_size=124, mode="inference", crs=None, res=None, transform=None):
        super(segDataset, self).__init__()
        self.root = root
        self.patch_size = patch_size
        self.mode = mode  # "train" eller "inference" (inference = testing uten annoteringsbilder)
        self.transform = transform
        self.crs = CRS.from_epsg(25833) if crs is None else crs  # Standard CRS hvis ikke spesifisert
        self.res = 0.25 if res is None else res  # Standard oppløsning hvis ikke spesifisert
        images_path = os.path.join(self.root, 'images', '*.tif')  # Sti til bildefiler
        
        self.IMG_NAMES = sorted(glob(images_path))  # Hent og sorter alle bildefiler
        print(f"Lastet {len(self.IMG_NAMES)} bilder fra {images_path}")

        # Lister for å lagre alle bildefeltene og deres posisjoner
        self.image_patches = []
        self.annotation_patches = [] if self.mode == "train" else None  # Bare nødvendig i treningsmodus
        self.positions = []
        self.image_indices = []  # Holder styr på bildeindekser
        self.image_shapes = []  # Lagrer originale bildestørrelser

        # Opprett felter ved initialisering
        self._create_patches()

    def _create_patches(self):
        for img_idx, img_path in enumerate(self.IMG_NAMES):
            # Sjekker bare for annotasjoner i treningsmodus
            if self.mode == "train":
                annotation_path = img_path.replace('images', 'annotations')  # Antatt struktur for annotasjoner
                if not os.path.exists(annotation_path):
                    print(f"Advarsel: Annotasjonsfil ikke funnet for {img_path}. Hopper over denne filen.")
                    continue

            # Laster inn bildet
            with rasterio.open(img_path) as src_image:
                image = src_image.read().transpose(1, 2, 0)  # Transponerer for å få riktig dimensjon
                image_meta = src_image.meta
                image_res = src_image.res
                image_transform = src_image.transform
                print(f"Bilde Metadata for {img_path}: CRS={image_meta['crs']}, Transform={image_transform}, Oppløsning={image_res}")
                self.image_shapes.append(image.shape[:2])  # Lagrer bildestørrelse (høyde, bredde)

            # Laster inn annotasjonen hvis i treningsmodus
            if self.mode == "train":
                with rasterio.open(annotation_path) as src_annotation:
                    annotation_res = src_annotation.res
                    annotation_transform = src_annotation.transform
                    # Sjekk om annotasjonen har samme CRS og oppløsning som bildet
                    #if annotation_res != image_res or annotation_transform != image_transform:
                        #print(f"Reprojiserer annotasjon for å matche CRS og oppløsning: {annotation_path}")
                        #annotation = src_annotation.read(
                            #out_shape=(
                                #src_annotation.count,
                                #int(image.shape[0]),
                                #int(image.shape[1])
                            #),
                            #resampling=Resampling.nearest
                        #)[0]
                    #else:
                    annotation = src_annotation.read(1)  # Les første band

            # Genererer felter for både bilde (og annotasjon hvis i treningsmodus)
            image_patches, annotation_patches, positions = self.crop_to_patches(
                image, annotation if self.mode == "train" else None
            )
            num_patches = len(image_patches)
            self.image_patches.extend(image_patches)
            if self.mode == "train":
                self.annotation_patches.extend(annotation_patches)
            self.positions.extend(positions)
            self.image_indices.extend([img_idx] * num_patches)  # Holder styr på bildeindekser

        print(f"Genererte {len(self.image_patches)} felter fra bilder i {self.root}")

    def crop_to_patches(self, image, annotation=None):
        patches = []
        annotation_patches = [] if annotation is not None else None
        positions = []

        height, width, _ = image.shape
        stride = self.patch_size

        for i in range(0, height - stride + 1, stride):
            for j in range(0, width - stride + 1, stride):
                image_patch = image[i:i+stride, j:j+stride]
                if annotation is not None:
                    annotation_patch = annotation[i:i+stride, j:j+stride]

                # Legger bare til felter med forventet størrelse
                if image_patch.shape[:2] == (stride, stride):
                    patches.append(image_patch)
                    positions.append((i, j))
                    if annotation is not None and annotation_patch.shape == (stride, stride):
                        annotation_patches.append(annotation_patch)
                else:
                    print(f"Hopper over felt ved ({i},{j}) på grunn av størrelsesmismatch")

        return patches, annotation_patches, positions if annotation_patches is not None else positions

    def __getitem__(self, idx): 
        image_patch = self.image_patches[idx]
        position = self.positions[idx]
        img_idx = self.image_indices[idx]

        # Transformer bildefelt til tensor
        image_patch = np.moveaxis(image_patch, -1, 0)  # Flytter kanaler til første dimensjon
        if self.mode == "train":
            annotation_patch = self.annotation_patches[idx]
            
            # Apply scaling for train mode
            image_patch = image_patch.astype(np.float32) * 0.70
            
            return (
                torch.tensor(image_patch, dtype=torch.float32),
                torch.tensor(annotation_patch, dtype=torch.int64),
                position,
                img_idx
            )
        else:
            return (
                torch.tensor(image_patch, dtype=torch.float32),
                position,
                img_idx
            )

    def __len__(self):
        return len(self.image_patches)
