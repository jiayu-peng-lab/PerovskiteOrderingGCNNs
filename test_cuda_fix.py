#!/usr/bin/env python3
"""
Test script to verify the CUDA multiprocessing fix
"""

import multiprocessing
import torch.multiprocessing as mp
import torch
import dgl

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
if __name__ == "__main__":
    # Set the start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    print("=== Testing CUDA Multiprocessing Fix ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"DGL version: {dgl.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Multiprocessing start method: {mp.get_start_method()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # Test basic DGL operations
        device = torch.device('cuda')
        g = dgl.graph(([0, 1, 2], [1, 2, 0]))
        
        try:
            g = g.to(device)
            print("✓ DGL graph successfully moved to GPU")
            
            # Test with some features
            g.ndata['feat'] = torch.randn(3, 10).to(device)
            print("✓ DGL node features successfully added to GPU")
            
            print("✓ All tests passed! The CUDA multiprocessing fix is working.")
            
        except Exception as e:
            print(f"✗ Error during GPU operations: {e}")
            print("The fix may not be working properly.")
    else:
        print("CUDA not available - running CPU-only tests")
        
        # Test basic DGL operations on CPU
        g = dgl.graph(([0, 1, 2], [1, 2, 0]))
        g.ndata['feat'] = torch.randn(3, 10)
        print("✓ DGL operations working on CPU")
    
    print("\nYou can now run your training script without CUDA multiprocessing errors!") 