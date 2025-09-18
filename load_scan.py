import matplotlib
matplotlib.use('Agg')
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom, binary_fill_holes
from skimage import measure, morphology

# --- FUNCTIONS ---
def convert_to_hu(image):
    intercept = -1024
    numpy_image = sitk.GetArrayFromImage(image)
    hu_image = numpy_image.astype(np.int16) + intercept
    return hu_image

def resample_to_isotropic(image, old_spacing, new_spacing=[1,1,1]):
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    resampled_image = zoom(image, real_resize_factor, order=1)
    return resampled_image, new_spacing

# --- FINAL, ROBUST LUNG SEGMENTATION FUNCTION ---
def segment_lung_mask(image_hu, verbose=False):
    """
    Creates a clean, binary mask of the lung parenchyma.
    """
    if verbose: print("Starting lung segmentation...")
    
    # Step 1: Create a binary mask of the patient's body
    body_mask = image_hu > -1000
    
    # Step 2: Fill holes in the body mask to get a solid representation
    solid_body = binary_fill_holes(body_mask)
    
    # Step 3: Invert the HU image and mask it with the solid body to get air inside
    inverted_hu = -(image_hu + 1024)
    air_inside_body = inverted_hu * solid_body
    
    # Step 4: Threshold to get the lung tissue and airways
    lung_airways_mask = air_inside_body > -400
    if verbose: print(f"Step 4 (Isolate Lungs/Airways): Found {np.sum(lung_airways_mask)} voxels.")
    
    # --- FINAL CLEANUP STEPS ---
    
    # Step 5: Erode the mask to disconnect airways from lungs
    # This removes thin connections.
    eroded_mask = morphology.binary_erosion(lung_airways_mask, np.ones((4,4,4)))
    
    # Step 6: Use connected components to keep only the largest regions (the lungs)
    labels = measure.label(eroded_mask)
    label_vals, label_counts = np.unique(labels, return_counts=True)
    label_counts = label_counts[label_vals != 0] # Exclude background
    label_vals = label_vals[label_vals != 0]
    
    if len(label_counts) > 1:
        top_two_labels = label_vals[np.argsort(label_counts)[::-1]][:2]
        lungs_only_mask = np.isin(labels, top_two_labels)
    elif len(label_counts) == 1:
        lungs_only_mask = labels > 0
    else: # No regions found after erosion
        lungs_only_mask = np.zeros_like(eroded_mask)

    if verbose: print(f"Step 6 (Keep Lungs): Found {np.sum(lungs_only_mask)} lung voxels after erosion.")

    # Step 7: Dilate the mask to restore the original lung size
    # This grows the lung regions back to their approximate original shape.
    final_mask = morphology.binary_dilation(lungs_only_mask, np.ones((5,5,5)))
    if verbose: print(f"Step 7 (Restore Size): Final mask has {np.sum(final_mask)} voxels.")

    return final_mask.astype(np.int16)


# --- SCRIPT LOGIC ---
subset_path = 'subset0/'
mhd_files = [f for f in os.listdir(subset_path) if f.endswith('.mhd')]
scan_path = os.path.join(subset_path, mhd_files[0])

itk_img = sitk.ReadImage(scan_path)
original_numpy_array = sitk.GetArrayFromImage(itk_img)
original_spacing = np.array(itk_img.GetSpacing())[[2,1,0]]

hu_numpy_array = convert_to_hu(itk_img)
resampled_array, _ = resample_to_isotropic(hu_numpy_array, original_spacing)

segmented_lungs_mask = segment_lung_mask(resampled_array, verbose=True)

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

slice_idx_orig = original_numpy_array.shape[0] // 2
slice_idx_resampled = resampled_array.shape[0] // 2

axes[0].imshow(original_numpy_array[slice_idx_orig, :, :], cmap='gray')
axes[0].set_title('1. Original Scan', fontsize=16)
axes[0].axis('off')

axes[1].imshow(resampled_array[slice_idx_resampled, :, :], cmap='gray', vmin=-1200, vmax=600)
axes[1].set_title('2. Resampled', fontsize=16)
axes[1].axis('off')

axes[2].imshow(segmented_lungs_mask[slice_idx_resampled, :, :], cmap='gray')
axes[2].set_title('3. Segmented Lung Mask', fontsize=16)
axes[2].axis('off')

lungs_only = np.where(segmented_lungs_mask == 1, resampled_array, -1024)
axes[3].imshow(lungs_only[slice_idx_resampled, :, :], cmap='gray', vmin=-1200, vmax=600)
axes[3].set_title('4. Segmented Lungs', fontsize=16)
axes[3].axis('off')

plt.subplots_adjust(top=0.9)
fig.suptitle('Preprocessing Steps', fontsize=20)
plt.savefig('full_preprocessing_visualization_final.png')
print("\nSuccessfully saved final visualization.")