import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# ===================================================================
# 1. THE DATASET CLASS (Your existing code)
# ===================================================================
class LungNoduleDataset(Dataset):
    """
    PyTorch Dataset for loading FULL 3D lung scans and their corresponding nodule masks.
    NOTE: This version is for verification and will likely cause memory errors during
    actual training on hardware with limited VRAM.
    """
    def __init__(self, data_dir, masks_dir):
        self.data_dir = data_dir
        self.masks_dir = masks_dir
        
        # We need to find scans that have a corresponding nodule mask
        mask_uids = {f.replace('_nodule_mask.npy', '') for f in os.listdir(masks_dir)}
        scan_files = {f.replace('_processed_lung.npy', '') for f in os.listdir(data_dir)}
        
        # Find the intersection of scans that are processed and have a mask
        self.available_uids = sorted(list(mask_uids.intersection(scan_files)))
        
        print(f"Found {len(self.available_uids)} scans with corresponding nodule masks.")

    def __len__(self):
        return len(self.available_uids)

    def __getitem__(self, idx):
        uid = self.available_uids[idx]
        
        scan_path = os.path.join(self.data_dir, f"{uid}_processed_lung.npy")
        mask_path = os.path.join(self.masks_dir, f"{uid}_nodule_mask.npy")
        
        scan = np.load(scan_path)
        mask = np.load(mask_path)
        
        # Convert the full arrays to PyTorch tensors and add a channel dimension
        scan_tensor = torch.from_numpy(scan).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return scan_tensor, mask_tensor

# ===================================================================
# 2. THE MODEL ARCHITECTURE (New code)
# ===================================================================
class ConvBlock(nn.Module):
    """Standard 3D Convolutional Block (Conv3D -> ReLU -> Conv3D -> ReLU)"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """Attention Gate to focus on relevant features"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet3D(nn.Module):
    """The full 3D Attention U-Net model"""
    def __init__(self, in_channels=1, out_channels=1):
        super(AttentionUNet3D, self).__init__()
        
        # Encoder
        self.Conv1 = ConvBlock(in_channels, 64)
        self.Pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv2 = ConvBlock(64, 128)
        self.Pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv3 = ConvBlock(128, 256)
        self.Pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv4 = ConvBlock(256, 512)
        self.Pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv5 = ConvBlock(512, 1024)

        # Decoder
        self.Up5 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.Att5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.Att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.Att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.Att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.UpConv2 = ConvBlock(128, 64)

        # Final Convolution
        self.Conv_1x1 = nn.Conv3d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder path
        x1 = self.Conv1(x)
        x2 = self.Pool1(x1)
        x2 = self.Conv2(x2)
        x3 = self.Pool2(x2)
        x3 = self.Conv3(x3)
        x4 = self.Pool3(x3)
        x4 = self.Conv4(x4)
        x5 = self.Pool4(x4)
        x5 = self.Conv5(x5)

        # Decoder path with attention gates
        d5 = self.Up5(x5)
        x4_att = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        x3_att = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        x2_att = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        x1_att = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv_1x1(d2)
        return torch.sigmoid(out) # Use sigmoid for binary segmentation

# ===================================================================
# 3. VERIFICATION SCRIPT (Updated to test the model)
# ===================================================================
if __name__ == '__main__':
    # --- Test the Dataset ---
    processed_data_dir = 'processed_data/'
    ground_truth_masks_dir = 'ground_truth_masks/'

    dataset = LungNoduleDataset(data_dir=processed_data_dir, 
                                masks_dir=ground_truth_masks_dir)

    print(f"\nTotal number of scans with nodules: {len(dataset)}")
    
    # --- Test the Model ---
    # Create an instance of the model
    model = AttentionUNet3D()
    
    # Test the model with a dummy input to check for errors
    # NOTE: The input size must be divisible by 16 (due to 4 pooling layers)
    # We use 96x96x96 as it's a common patch size and is divisible by 16.
    dummy_input = torch.randn(1, 1, 96, 96, 96) # (batch, channels, depth, height, width)
    output = model(dummy_input)
    
    print("\n--- Model Test ---")
    print(f"Model created successfully.")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")
    
    # Check if input and output shapes match
    assert dummy_input.shape == output.shape, "Model output shape does not match input shape!"
    print("✅ Model input and output shapes match.")