#!/usr/bin/env python3
"""
Test script to verify DGL CUDA support is working for ALIGNN training
"""

import os
import torch
import dgl
import wandb

# Set environment variables for DGL GPU usage
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def test_dgl_cuda():
    """Test DGL CUDA functionality"""
    print("=== Testing DGL CUDA Support ===")
    print(f"DGL version: {dgl.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test basic DGL operations on GPU
        device = torch.device('cuda')
        g = dgl.graph(([0, 1, 2], [1, 2, 0]))
        g = g.to(device)
        print("✓ DGL graph successfully moved to GPU")
        
        # Test node features on GPU
        g.ndata['feat'] = torch.randn(3, 10).to(device)
        print("✓ DGL node features successfully added to GPU")
        
        # Test edge features on GPU
        g.edata['weight'] = torch.randn(3).to(device)
        print("✓ DGL edge features successfully added to GPU")
        
        print("✓ All DGL CUDA tests passed!")
        return True
    else:
        print("✗ CUDA not available")
        return False

def test_alignn_imports():
    """Test ALIGNN-related imports"""
    print("\n=== Testing ALIGNN Imports ===")
    try:
        from models.PerovskiteOrderingGCNNs_alignn.alignn.dataset import get_torch_dataset
        print("✓ ALIGNN dataset module imported successfully")
        
        from processing.dataloader.dataloader_gpu_optimized import get_alignn_dataloader_gpu_optimized
        print("✓ ALIGNN GPU-optimized dataloader imported successfully")
        
        print("✓ All ALIGNN imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Main test function"""
    print("Testing DGL CUDA fix for ALIGNN training...")
    
    # Test DGL CUDA support
    dgl_ok = test_dgl_cuda()
    
    # Test ALIGNN imports
    alignn_ok = test_alignn_imports()
    
    if dgl_ok and alignn_ok:
        print("\n🎉 All tests passed! DGL CUDA issue is fixed.")
        print("You can now run ALIGNN training with GPU support.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return dgl_ok and alignn_ok

if __name__ == "__main__":
    main() 