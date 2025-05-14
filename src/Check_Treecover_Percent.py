import os
import glob
import numpy as np
import rasterio

def calculate_class_percentages(annotation_dir, class_labels=None):
    if class_labels is None:
        class_labels = {0: 'Annet', 1: 'Tredekke'}
    
    # Hent en liste over alle .tif-filer i annotasjonskatalogen
    annotation_files = glob.glob(os.path.join(annotation_dir, '*.tif'))
    print(f"Fant {len(annotation_files)} annotasjonsfiler i {annotation_dir}")
    
    # Initialiser variabler for å holde oversikt over totale klasseantall
    overall_class_counts = {label: 0 for label in class_labels.keys()}
    overall_total_pixels = 0
    
    # Prosesser hver annotasjonsfil
    for annotation_file in annotation_files:
        with rasterio.open(annotation_file) as src:
            annotation_data = src.read(1)  # Les den første bandet
            # Flatt arrayet til 1D for enklere behandling
            annotation_data = annotation_data.flatten()
            
            # Få unike klasser og deres antall
            unique, counts = np.unique(annotation_data, return_counts=True)
            class_counts = dict(zip(unique, counts))
            
            total_pixels = counts.sum()
            overall_total_pixels += total_pixels
    
            # Oppdater totale klasseantall
            for cls in class_labels.keys():
                count = class_counts.get(cls, 0)
                overall_class_counts[cls] += count
    
            # Beregn prosentandeler og inkluder pikselantall for denne filen
            print(f"\nFil: {os.path.basename(annotation_file)}")
            print(f"  Totalt antall piksler: {total_pixels}")
            for cls in class_labels.keys():
                count = class_counts.get(cls, 0)
                percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
                print(f"  Klasse '{class_labels[cls]}' ({cls}): {count} piksler ({percentage:.2f}%)")
    
    # Beregn totale prosentandeler og inkluder pikselantall
    print("\nTotale Klasseprosentandeler og Pikselantall:")
    print(f"  Totalt antall piksler: {overall_total_pixels}")
    for cls in class_labels.keys():
        count = overall_class_counts[cls]
        percentage = (count / overall_total_pixels) * 100 if overall_total_pixels > 0 else 0
        print(f"  Klasse '{class_labels[cls]}' ({cls}): {count} piksler ({percentage:.2f}%)")

# Spesifiser katalogen som inneholder dine annotasjons TIFF-filer
annotation_directory = r'PATH_TIL_ANNOTASJON_DIR'

# Valgfritt, definer klassene hvis de er forskjellige
class_labels = {
    0: 'Annet',
    1: 'Tredekke'
}

calculate_class_percentages(annotation_directory, class_labels)
