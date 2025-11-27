import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models.model import SwinAD2Net

def verify():
    print("Initializing model...")
    model = SwinAD2Net(num_classes=2, embed_dim=96, image_size=224, patch_size_embed=4, growth_rate=32)
    model.eval()
    
    print("Creating dummy input...")
    x = torch.randn(2, 3, 224, 224)
    
    print("Running forward pass...")
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print("Verification successful!")

if __name__ == "__main__":
    verify()
