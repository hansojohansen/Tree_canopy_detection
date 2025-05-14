import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Specify the path to your TIFF file
tif_file_path = r'C:\Users\milge\OneDrive\Dokumenter\Johansen\Bachelor Oppgave\Database\testing\annotations\Elvegaardsmoen2021.tif'

# Open the TIFF file
with rasterio.open(tif_file_path) as src:
    data = src.read(1)  # Read the first band

# Handle no-data values if necessary
no_data_value = src.nodata
if no_data_value is not None:
    data = np.ma.masked_equal(data, no_data_value)

# Display the data with a green color map
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='Greens', vmin=0, vmax=1)
plt.colorbar()
plt.title('TIFF File Display - Green Color Map')
plt.xlabel('Column')
plt.ylabel('Row')
plt.show()
