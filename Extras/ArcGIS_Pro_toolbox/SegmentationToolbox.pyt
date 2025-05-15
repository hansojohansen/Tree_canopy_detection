import os
'''If CUDA error codes occur, one possible solution is to specify the file paths for PROJ_LIB
to the directory containing proj.db'''
#os.environ['PROJ_LIB'] = r"PATH_TO_PROJ_LIB"
#os.add_dll_directory(r"PATH_TO_CUDA_BIN")
#os.add_dll_directory(r"PATH_TO_CUDA_LIBNVVP")

import arcpy
import torch
import torch.nn as nn
import torch.nn.functional as F  # Sørg for at denne importen er inkludert
import numpy as np
from torch.utils.data import DataLoader
import rasterio
from rasterio.enums import Resampling
from rasterio.crs import CRS
from glob import glob

class Toolbox:
    def __init__(self):
        """Define the toolbox."""
        self.label = "Segmentation Toolbox"
        self.alias = "segmentation_toolbox"
        self.tools = [SegmentationTool]

class SegmentationTool:
    def __init__(self):
        """Defines the toolbox."""
        self.label = "Image Segmentation Tool"
        self.description = "Performs image segmentation using a trained UNet model."

    def getParameterInfo(self):
        """Define the tool's parameters."""
        param0 = arcpy.Parameter(
            displayName="Input Data Directory",
            name="input_data_dir",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")

        param1 = arcpy.Parameter(
            displayName="Trained Model File",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input")

        param2 = arcpy.Parameter(
            displayName="Output Directory",
            name="output_dir",
            datatype="DEFolder",
            parameterType="Required",
            direction="Output")

        param3 = arcpy.Parameter(
            displayName="Tile Size",
            name="tile_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        param3.value = 512  # Here you can set the default value

        param4 = arcpy.Parameter(
            displayName="Use Bilinear Upscaling",
            name="upscaling_method",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input")
        param4.value = True  # Here you can set the default value

        param5 = arcpy.Parameter(
            displayName="Probability Threshold",
            name="probability_threshold",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")
        param5.value = 0.5  # Here you can set the default value


        return [param0, param1, param2, param3, param4, param5]

    def isLicensed(self):
        return True

    def execute(self, parameters, messages):
        input_data_dir = parameters[0].valueAsText
        model_file = parameters[1].valueAsText
        output_dir = parameters[2].valueAsText
        tile_size = parameters[3].value
        upscaling_method = parameters[4].value
        threshold = parameters[5].value

        arcpy.AddMessage(f"Input Data Directory: {input_data_dir}")
        arcpy.AddMessage(f"Model File: {model_file}")
        arcpy.AddMessage(f"Output Directory: {output_dir}")
        arcpy.AddMessage(f"Tile Size: {tile_size}")
        arcpy.AddMessage(f"Upscaling Method (bilinear): {upscaling_method}")
        arcpy.AddMessage(f"Probability Threshold: {threshold}")

        class segDataset(torch.utils.data.Dataset):
            def __init__(self, root, patch_size=124, crs=None, res=None):
                super(segDataset, self).__init__()
                self.root = root
                self.patch_size = patch_size

                # Try to set CRS using EPSG code, fallback to WKT, and finally a default
                try:
                    self.crs = CRS.from_epsg(25833) if crs is None else crs
                    arcpy.AddMessage("Uses CRS from EPSG: 25833")
                except rasterio.errors.CRSError:
                    arcpy.AddMessage("EPSG-kode failed, trying WKT.")
                    try:
                        self.crs = CRS.from_wkt('PROJCS["ETRS89 / UTM zone 33N", ...]')
                        arcpy.AddMessage("Uses CRS from WKT")
                    except rasterio.errors.CRSError:
                        arcpy.AddMessage("WKT parsing failed, using standard CRS")
                        self.crs = CRS.from_epsg(4326)  # Fallback to WGS84

                self.res = 0.25 if res is None else res
                images_path = os.path.join(self.root, '*.tif')
                self.IMG_NAMES = sorted(glob(images_path))

                arcpy.AddMessage(f"Loaded {len(self.IMG_NAMES)} images from {images_path}")
                self.image_patches = []
                self.positions = []
                self.image_indices = []
                self.image_shapes = []

                self._create_patches()

            def _create_patches(self):
                for img_idx, img_path in enumerate(self.IMG_NAMES):
                    with rasterio.open(img_path) as src_image:
                        image = src_image.read().transpose(1, 2, 0)
                        self.image_shapes.append(image.shape[:2])
                        image_patches, positions = self.crop_to_patches(image)
                        self.image_patches.extend(image_patches)
                        self.positions.extend(positions)
                        self.image_indices.extend([img_idx] * len(image_patches))

                arcpy.AddMessage(f"Generated {len(self.image_patches)} tiles.")

            def crop_to_patches(self, image):
                patches = []
                positions = []
                height, width, _ = image.shape
                stride = self.patch_size

                for i in range(0, height - stride + 1, stride):
                    for j in range(0, width - stride + 1, stride):
                        patch = image[i:i+stride, j:j+stride]
                        if patch.shape[:2] == (stride, stride):
                            patches.append(patch)
                            positions.append((i, j))
                return patches, positions

            def __getitem__(self, idx):
                patch = np.moveaxis(self.image_patches[idx], -1, 0)
                return torch.tensor(patch).float(), self.positions[idx], self.image_indices[idx]

            def __len__(self):
                return len(self.image_patches)

        # Model classes similar to training script
        class DoubleConv(nn.Module):
            """(Convolution => [BN] => ReLU) * 2"""
            def __init__(self, in_channels, out_channels, mid_channels=None):
                super().__init__()
                if not mid_channels:
                    mid_channels = out_channels
                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            def forward(self, x):
                return self.double_conv(x)

        class Down(nn.Module):
            """Downscaling with maxpool followed by double convolution"""
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.maxpool_conv = nn.Sequential(
                    nn.MaxPool2d(2),  # Halves the spatial dimensions
                    DoubleConv(in_channels, out_channels)
                )
            def forward(self, x):
                return self.maxpool_conv(x)

        class Up(nn.Module):
            """Upscaling followed by double convolution"""
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()
                if bilinear:
                    # Use bilinear upscaling
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                    self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
                else:
                    # Use transposed convolution
                    self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                    self.conv = DoubleConv(in_channels, out_channels)
            def forward(self, x1, x2):
                x1 = self.up(x1)
                # Resize if necessary
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                # Concatenate along the channel axis
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)

        class OutConv(nn.Module):
            """Output convolution layer"""
            def __init__(self, in_channels, out_channels):
                super(OutConv, self).__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            def forward(self, x):
                return self.conv(x)

        class UNet(nn.Module): # Must be the same as what the model was trained on
            def __init__(self, n_channels, n_classes, bilinear=True):
                super(UNet, self).__init__()
                self.n_channels = n_channels
                self.n_classes = n_classes
                self.bilinear = bilinear

                self.inc = DoubleConv(n_channels, 64)
                self.down1 = Down(64, 128)
                self.down2 = Down(128, 256)
                self.down3 = Down(256, 512)
                factor = 2 if bilinear else 1
                self.down4 = Down(512, 1024 // factor)
                self.up1 = Up(1024, 512 // factor, bilinear)
                self.up2 = Up(512, 256 // factor, bilinear)
                self.up3 = Up(256, 128 // factor, bilinear)
                self.up4 = Up(128, 64, bilinear)
                self.outc = OutConv(64, n_classes)

            def forward(self, x):
                x1 = self.inc(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
                x = self.up1(x5, x4)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
                logits = self.outc(x)
                return logits

        # Unit configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arcpy.AddMessage(f"Uses device: {device}")

        # Dataset and DataLoader
        testing_dataset = segDataset(root=input_data_dir, patch_size=tile_size)
        # Custom collate function to handle positions
        def custom_collate_fn(batch):
            if len(batch) == 0:
                return None
            images, positions, image_indices = [], [], []
            for item in batch:
                images.append(item[0])
                positions.append(item[1])
                image_indices.append(item[2])
            images = torch.stack(images, 0)
            return images, positions, image_indices

        testing_dataloader = DataLoader(
            testing_dataset,
            batch_size=1,  # You can adjust batch size if needed
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )

        # Model instantiation
        model = UNet(n_channels=3, n_classes=2, bilinear=upscaling_method).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        arcpy.AddMessage("Model loaded and ready for inference")

        # Inference code
        with torch.no_grad():
            predictions_per_image = {}
            for batch in testing_dataloader:
                if batch is None:
                    continue

                inputs, positions, image_indices = batch

                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)  # Calculate softmax probabilities
                class1_probs = probs[:, 1, :, :]       # Extract probabilities for class 1

                # Apply threshold to obtain binary predictions
                preds = (class1_probs > threshold).long()
                preds = preds.cpu().numpy()

                batch_size_actual = inputs.size(0)
                for i in range(batch_size_actual):
                    pred_patch = preds[i]
                    x, y = positions[i]
                    img_idx = image_indices[i]

                    if img_idx not in predictions_per_image:
                        height, width = testing_dataset.image_shapes[img_idx]
                        predictions_per_image[img_idx] = np.zeros((height, width), dtype=np.uint8)

                    predictions_per_image[img_idx][x:x+tile_size, y:y+tile_size] = pred_patch

        # Save each reconstructed mask as a georeferenced raster layer
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
            output_path = os.path.join(output_dir, output_filename)

            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(reconstructed_mask.astype(rasterio.uint8), 1)

            arcpy.AddMessage(f"Predicted masks saved to {output_path}")

        arcpy.AddMessage("Inference completed successfully.")
        return
