import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import time

# --- Helper Functions (from previous script) ---
def world_to_voxel(world_coords, origin, spacing):
    stretched_voxel_coords = np.absolute(world_coords - origin)
    voxel_coords = stretched_voxel_coords / spacing
    return (int(voxel_coords[2]), int(voxel_coords[1]), int(voxel_coords[0]))

def find_scan_path(series_uid, scan_folder):
    for file_name in os.listdir(scan_folder):
        if file_name.startswith(series_uid) and file_name.endswith('.mhd'):
            return os.path.join(scan_folder, file_name)
    return None

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    # Define paths
    annotations_path = 'data/annotations.csv'
    raw_scan_folder = 'data/subset0/'
    processed_data_folder = 'Preprocessed_Data_0/processed_data_subset0/'
    output_mask_folder = 'ground_truth_masks/'

    # Create output folder if it doesn't exist
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)
        print(f"Created output folder: {output_mask_folder}")

    # Load annotations
    df_annos = pd.read_csv(annotations_path)

    # Get UIDs of scans we have already processed
    processed_scans_uids = {f.split('_')[0] for f in os.listdir(processed_data_folder) if f.endswith('_lung_mask.npy')}
    print(f"Found {len(processed_scans_uids)} processed scans.")

    # Filter annotations for the scans we have
    df_subset_annos = df_annos[df_annos['seriesuid'].isin(processed_scans_uids)].copy()
    print(f"Found {len(df_subset_annos)} nodules in the available processed scans.")

    # Loop through each scan that has annotations
    for series_uid, scan_annotations in df_subset_annos.groupby('seriesuid'):
        start_time = time.time()
        print(f"\n--- Creating ground truth mask for: {series_uid} ---")
        
        # --- Step 1: Get metadata and shapes ---
        
        # Load the processed lung mask to get the target shape of our ground truth mask
        processed_mask_path = os.path.join(processed_data_folder, f"{series_uid}_lung_mask.npy")
        processed_mask = np.load(processed_mask_path)
        resampled_shape = processed_mask.shape

        # Load original scan metadata to map coordinates
        raw_scan_path = find_scan_path(series_uid, raw_scan_folder)
        if not raw_scan_path:
            print(f"!!! Warning: Could not find raw scan for {series_uid}. Skipping.")
            continue
            
        itk_img = sitk.ReadImage(raw_scan_path)
        origin = np.array(itk_img.GetOrigin())
        original_spacing = np.array(itk_img.GetSpacing())
        
        # Calculate the resampling factor used during preprocessing
        # Note: Original spacing is (x,y,z), numpy shape is (z,y,x)
        original_shape = np.array(itk_img.GetSize())[[2,1,0]]
        resample_factor = resampled_shape / original_shape

        # --- Step 2: Create a blank mask and draw nodules on it ---
        
        nodule_mask = np.zeros(resampled_shape, dtype=np.int16)

        for index, row in scan_annotations.iterrows():
            world_coords = np.array([row['coordX'], row['coordY'], row['coordZ']])
            diameter_mm = row['diameter_mm']

            # Convert world coords to original voxel coords
            original_voxel_coords = world_to_voxel(world_coords, origin, original_spacing)
            
            # Convert original voxel coords to resampled voxel coords
            resampled_voxel_coords = np.round(original_voxel_coords * resample_factor).astype(int)
            z, y, x = resampled_voxel_coords
            
            # Draw a sphere for the nodule
            radius_in_voxels = int(np.ceil((diameter_mm / 2))) # Radius in 1x1x1mm voxel space
            
            for i in range(z - radius_in_voxels, z + radius_in_voxels + 1):
                for j in range(y - radius_in_voxels, y + radius_in_voxels + 1):
                    for k in range(x - radius_in_voxels, x + radius_in_voxels + 1):
                        # Check if the point is within the sphere and the array bounds
                        if (i-z)**2 + (j-y)**2 + (k-x)**2 <= radius_in_voxels**2:
                            if 0 <= i < resampled_shape[0] and 0 <= j < resampled_shape[1] and 0 <= k < resampled_shape[2]:
                                nodule_mask[i, j, k] = 1

            print(f"  > Drawn nodule at resampled coords (z,y,x): {tuple(resampled_voxel_coords)}")

        # --- Step 3: Save the final mask for this scan ---
        
        output_path = os.path.join(output_mask_folder, f"{series_uid}_nodule_mask.npy")
        np.save(output_path, nodule_mask)
        
        end_time = time.time()
        print(f"  Saved ground truth mask to {output_path} in {end_time - start_time:.2f} seconds.")