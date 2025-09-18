import SimpleITK as sitk
import numpy as np
import os
from scipy.ndimage import zoom, binary_fill_holes
from skimage import measure, morphology
import time # To track how long it takes

# --- ALL PREPROCESSING AND SEGMENTATION FUNCTIONS ---
# (Copy and paste all the functions we perfected:
# convert_to_hu, resample_to_isotropic, and the final segment_lung_mask)

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

def segment_lung_mask(image_hu, verbose=False):
    if verbose: print("Starting lung segmentation...")
    body_mask = image_hu > -1000
    solid_body = binary_fill_holes(body_mask)
    inverted_hu = -(image_hu + 1024)
    air_inside_body = inverted_hu * solid_body
    lung_airways_mask = air_inside_body > -400
    if verbose: print(f"Step 4 (Isolate Lungs/Airways): Found {np.sum(lung_airways_mask)} voxels.")
    eroded_mask = morphology.binary_erosion(lung_airways_mask, np.ones((4,4,4)))
    labels = measure.label(eroded_mask)
    label_vals, label_counts = np.unique(labels, return_counts=True)
    label_counts = label_counts[label_vals != 0]
    label_vals = label_vals[label_vals != 0]
    if len(label_counts) > 1:
        top_two_labels = label_vals[np.argsort(label_counts)[::-1]][:2]
        lungs_only_mask = np.isin(labels, top_two_labels)
    elif len(label_counts) == 1:
        lungs_only_mask = labels > 0
    else:
        lungs_only_mask = np.zeros_like(eroded_mask)
    if verbose: print(f"Step 6 (Keep Lungs): Found {np.sum(lungs_only_mask)} lung voxels after erosion.")
    final_mask = morphology.binary_dilation(lungs_only_mask, np.ones((5,5,5)))
    if verbose: print(f"Step 7 (Restore Size): Final mask has {np.sum(final_mask)} voxels.")
    return final_mask.astype(np.int16)


# --- MAIN PROCESSING SCRIPT ---
def preprocess_all_scans(input_folder, output_folder):
    """
    Loops through all .mhd scans in the input folder, preprocesses them,
    and saves the processed lung images and masks as .npy files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    mhd_files = [f for f in os.listdir(input_folder) if f.endswith('.mhd')]
    total_files = len(mhd_files)
    print(f"Found {total_files} scans to process.")

    for i, file_name in enumerate(mhd_files):
        start_time = time.time()
        print(f"\n--- Processing scan {i+1}/{total_files}: {file_name} ---")
        
        try:
            # 1. Load Scan
            scan_path = os.path.join(input_folder, file_name)
            itk_img = sitk.ReadImage(scan_path)
            
            # 2. Preprocess (HU, Resample)
            original_spacing = np.array(itk_img.GetSpacing())[[2,1,0]]
            hu_numpy_array = convert_to_hu(itk_img)
            resampled_array, _ = resample_to_isotropic(hu_numpy_array, original_spacing)
            
            # 3. Segment Lungs
            segmented_mask = segment_lung_mask(resampled_array)
            
            # 4. Apply Mask
            processed_lung_image = resampled_array * segmented_mask
            
            # 5. Save the results
            # The file name is based on the series UID from the DICOM standard
            series_uid = os.path.splitext(file_name)[0]
            
            # Save the processed lung image
            np.save(os.path.join(output_folder, f"{series_uid}_processed_lung.npy"), processed_lung_image)
            
            # Save the lung mask
            np.save(os.path.join(output_folder, f"{series_uid}_lung_mask.npy"), segmented_mask)

            end_time = time.time()
            print(f"Finished processing in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"!!! FAILED to process {file_name}. Error: {e}")

# --- RUN THE SCRIPT ---
if __name__ == '__main__':
    input_scan_folder = 'data/subset0/'
    output_processed_folder = 'processed_data/'
    
    preprocess_all_scans(input_scan_folder, output_processed_folder)