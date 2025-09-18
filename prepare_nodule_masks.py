import numpy as np
import pandas as pd
import SimpleITK as sitk
import os

def world_to_voxel(world_coords, origin, spacing):
    stretched_voxel_coords = np.absolute(world_coords - origin)
    voxel_coords = stretched_voxel_coords / spacing
    return (int(voxel_coords[2]), int(voxel_coords[1]), int(voxel_coords[0]))

def get_scan_metadata(scan_path):
    itk_img = sitk.ReadImage(scan_path)
    origin = np.array(itk_img.GetOrigin())
    spacing = np.array(itk_img.GetSpacing())
    return origin, spacing

def find_scan_path(series_uid, scan_folder):
    for file_name in os.listdir(scan_folder):
        if file_name.startswith(series_uid) and file_name.endswith('.mhd'):
            return os.path.join(scan_folder, file_name)
    return None

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    annotations_path = 'data/annotations.csv'
    raw_scan_folder = 'data/subset0/'
    
    # Load the annotations CSV file
    df_annos = pd.read_csv(annotations_path)
    print("Successfully loaded annotations.")

    # --- NEW LOGIC: PROCESS ONLY AVAILABLE SCANS ---

    # 1. Get the list of scan UIDs that are actually in our folder
    available_scans_uids = {os.path.splitext(f)[0] for f in os.listdir(raw_scan_folder) if f.endswith('.mhd')}
    print(f"Found {len(available_scans_uids)} scans in '{raw_scan_folder}'.")

    # 2. Filter the annotations to only include those for our available scans
    df_subset_annos = df_annos[df_annos['seriesuid'].isin(available_scans_uids)].copy()
    print(f"Found {len(df_subset_annos)} nodules corresponding to the available scans.")

    # 3. Loop through the filtered annotations and process them
    print("\n--- Processing available scans and their nodules ---")
    # Group annotations by scan UID
    for series_uid, scan_annotations in df_subset_annos.groupby('seriesuid'):
        scan_path = find_scan_path(series_uid, raw_scan_folder)
        
        if scan_path:
            print(f"\nProcessing Scan: {series_uid}")
            origin, spacing = get_scan_metadata(scan_path)
            
            # Loop through each nodule found in this scan
            for index, row in scan_annotations.iterrows():
                world_x = row['coordX']
                world_y = row['coordY']
                world_z = row['coordZ']
                
                # Convert world coordinates to original voxel coordinates
                voxel_coords = world_to_voxel((world_x, world_y, world_z), origin, spacing)
                
                print(f"  - Nodule {index}:")
                print(f"    World Coords (x,y,z): ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})")
                print(f"    Voxel Coords (z,y,x): {voxel_coords}")
        else:
            print(f"!!! Warning: Could not find file for {series_uid}, though it was expected.")