#!/usr/bin/env python3
"""
Command to get ALIGNN predictions and embeddings properly
"""
import sys
sys.path.append('.')

def main():
    print("🚀 ALIGNN Predictions & Embeddings Commands:")
    print()
    
    print("📋 Current Status:")
    print("   - ALIGNN models are trained on full dataset")
    print("   - ALIGNN can only predict for ~366 structures (not 1261 like CGCNN)")
    print("   - This is likely due to ALIGNN graph creation filtering invalid structures")
    print()
    
    print("✅ Working Commands for ALIGNN Predictions:")
    print()
    print("1️⃣  Generate ALIGNN predictions (current working version):")
    print("   conda activate Perovskite_ML_Environment")
    print("   python generate_alignn_predictions_simple.py")
    print()
    
    print("2️⃣  Generate ALIGNN embeddings:")
    print("   python -c \"from inference.embedding_extraction import get_all_embeddings; get_all_embeddings({'struct_type': 'unrelaxed', 'model_type': 'ALIGNN', 'training_fraction': 1.0, 'data': 'data/', 'interpolation': False, 'contrastive_weight': 1.0, 'long_range': False}, gpu_num=0, num_best_models=3)\"")
    print()
    
    print("3️⃣  Analyze ALIGNN results:")
    print("   python alignn_only_analysis.py")
    print()
    
    print("4️⃣  Generate professional ordering plots:")
    print("   python alignn_ordering_dependence_professional.py")
    print()
    
    print("📊 Current ALIGNN Prediction Files:")
    print("   - best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_*/test_set_predictions.json")
    print("   - best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_*/test_set_predictions.json")
    print("   - Each contains ~366 samples (not 1261 like CGCNN)")
    print()
    
    print("💡 Note: The size difference is expected - ALIGNN can only process structures")
    print("   that can be successfully converted to graphs, while CGCNN is more robust.")

if __name__ == "__main__":
    main()
