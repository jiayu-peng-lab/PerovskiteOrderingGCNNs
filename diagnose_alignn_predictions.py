#!/usr/bin/env python3
"""
Diagnose why ALIGNN predictions are incomplete
"""
import sys
sys.path.append('.')

import pandas as pd
import json

def main():
    print("🔍 Diagnosing ALIGNN prediction issue...")
    
    # Check test data size
    try:
        test_data = pd.read_json('data/test_set.json')
        print(f"📊 Test data size: {len(test_data)} samples")
        print(f"📋 Test data columns: {test_data.columns.tolist()}")
        print(f"🔢 Test data indices: {test_data.index.min()} to {test_data.index.max()}")
        
        # Check if there's a crystal_id column
        if 'crystal_id' in test_data.columns:
            print(f"💎 Crystal IDs range: {test_data['crystal_id'].min()} to {test_data['crystal_id'].max()}")
            print(f"🔢 Unique crystal IDs: {test_data['crystal_id'].nunique()}")
        else:
            print("⚠️  No crystal_id column found in test data")
            
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Check CGCNN prediction file
    try:
        cgcnn_data = pd.read_json('best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_0/test_set_predictions.json')
        print(f"\n📊 CGCNN predictions: {len(cgcnn_data)} samples")
        print(f"📋 CGCNN columns: {cgcnn_data.columns.tolist()}")
        
        if 'crystal_id' in cgcnn_data.columns:
            print(f"💎 CGCNN crystal IDs range: {cgcnn_data['crystal_id'].min()} to {cgcnn_data['crystal_id'].max()}")
        else:
            print("⚠️  No crystal_id in CGCNN predictions")
            
    except Exception as e:
        print(f"❌ Error loading CGCNN predictions: {e}")
    
    # Check ALIGNN prediction file  
    try:
        alignn_data = pd.read_json('best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_0/test_set_predictions.json')
        print(f"\n📊 ALIGNN predictions: {len(alignn_data)} samples")
        print(f"📋 ALIGNN columns: {alignn_data.columns.tolist()}")
        
        if 'crystal_id' in alignn_data.columns:
            print(f"💎 ALIGNN crystal IDs range: {alignn_data['crystal_id'].min()} to {alignn_data['crystal_id'].max()}")
        else:
            print("⚠️  No crystal_id in ALIGNN predictions")
            
        print(f"\n🔍 ALIGNN prediction sample:")
        print(alignn_data.head(3))
        
    except Exception as e:
        print(f"❌ Error loading ALIGNN predictions: {e}")
    
    # Check ALIGNN model files
    print(f"\n🔍 Checking ALIGNN model structure...")
    try:
        with open('best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_0/training_results.json', 'r') as f:
            training_results = json.load(f)
            print(f"📈 ALIGNN training results keys: {list(training_results.keys())}")
            if 'test_mae' in training_results:
                print(f"📊 ALIGNN test MAE: {training_results['test_mae']}")
            if 'val_loss' in training_results:
                print(f"📊 ALIGNN validation loss: {training_results['val_loss']}")
                
    except Exception as e:
        print(f"❌ Error loading ALIGNN training results: {e}")

    print(f"\n💡 Analysis:")
    print(f"   - If test data has 1261 samples and CGCNN predicts 1261 but ALIGNN only predicts ~10")
    print(f"   - Then ALIGNN model loading or data processing is failing for most samples")
    print(f"   - This suggests an issue in the ALIGNN dataloader or model inference")

if __name__ == "__main__":
    main()
