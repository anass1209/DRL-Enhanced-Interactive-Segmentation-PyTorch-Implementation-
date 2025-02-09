import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Pour la barre de progression

# Dossier contenant les fichiers Nifti (.nii ou .nii.gz)
nifti_folder = r"C:\Users\Anass\Downloads"

# Dossier de sortie pour les images PNG
output_folder = "output_png_images"
os.makedirs(output_folder, exist_ok=True)

# Obtenir tous les fichiers Nifti du dossier
nifti_files = [f for f in os.listdir(nifti_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]

def convert_nifti_to_png(nifti_path, output_dir, num_slices=60):
    """Convertit un fichier Nifti en PNG et enregistre seulement les 60 premières slices."""
    nifti_img = nib.load(nifti_path)
    nifti_data = nifti_img.get_fdata()  # Convertir en tableau NumPy
    filename = os.path.basename(nifti_path).replace(".nii.gz", "").replace(".nii", "")

    # Créer un sous-dossier pour chaque fichier Nifti
    nifti_output_folder = os.path.join(output_dir, filename)
    os.makedirs(nifti_output_folder, exist_ok=True)

    # Parcourir les 60 premières slices (ou le max disponible)
    max_slices = min(nifti_data.shape[2], num_slices)
    for i in range(max_slices):  
        slice_img = nifti_data[:, :, i]
        slice_img = np.rot90(slice_img)  # Rotation pour un affichage correct
        output_path = os.path.join(nifti_output_folder, f"slice_{i:03d}.png")
        plt.imsave(output_path, slice_img, cmap='gray')

# Conversion avec une barre de progression
for nifti_file in tqdm(nifti_files, desc="Conversion en PNG"):
    nifti_path = os.path.join(nifti_folder, nifti_file)
    convert_nifti_to_png(nifti_path, output_folder, num_slices=60)

print(f"Conversion terminée ! Les 60 premières slices sont enregistrées dans '{output_folder}'")
