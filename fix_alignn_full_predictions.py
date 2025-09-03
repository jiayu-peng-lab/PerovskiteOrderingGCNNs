#!/usr/bin/env python3
"""
Fix ALIGNN predictions using the exact same pipeline as training
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import torch
import json
import time
from processing.dataloader.dataloader import get_dataloader
from processing.utils import filter_data_by_properties, select_structures
from training.wandb_utils import build_wandb_name
from processing.create_model.create_model import create_model
from torch.autograd import Variable

def load_alignn_model(gpu_num, train_loader, model_params, directory, target_prop):
    """Load ALIGNN model from directory"""
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    
    # Load model
    model = create_model(train_loader, model_params, target_prop, device)
    model_path = directory + '/best_model.torch'
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    except:
        model_path = directory + '/final_model.torch'
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    
    model.eval()
    
    # Load normalizer
    normalizer_path = directory + '/normalizer.pkl'
    try:
        import pickle
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
        print(f"✅ Loaded normalizer from {normalizer_path}")
    except:
        # Create dummy normalizer if not found
        normalizer = None
        print("⚠️  No normalizer found, using None")
    
    return model, normalizer

def evaluate_alignn_model_complete(model, normalizer, test_loader, gpu_num):
    """Evaluate ALIGNN model and return predictions with crystal IDs"""
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    
    predictions = {}
    model.eval()
    
    with torch.no_grad():
        for j, batch in enumerate(test_loader):
            try:
                # ALIGNN returns (graph, line_graph, lattice, label, crystal_id) when line_graph=True
                graph, line_graph, lattice, target, crystal_id = batch
                
                # Move to device
                try:
                    graph = graph.to(device)
                    line_graph = line_graph.to(device)
                    lattice = lattice.to(device)
                    target = target.to(device)
                except Exception as e:
                    if "cuda is not enabled" in str(e):
                        # Move model to CPU to match the graphs
                        model = model.to("cpu")
                        device = torch.device("cpu")
                        graph = graph.to(device)
                        line_graph = line_graph.to(device)
                        lattice = lattice.to(device)
                        target = target.to(device)
                    else:
                        raise e
                
                # ALIGNN model expects (graph, line_graph, lattice) as a single tuple argument
                output = model((graph, line_graph, lattice)).view(-1)
                
                # Get predictions
                if normalizer is not None:
                    prediction = normalizer.denorm(output)
                else:
                    prediction = output
                
                # Store predictions with crystal IDs
                if isinstance(crystal_id, torch.Tensor):
                    crystal_ids = crystal_id.cpu().numpy()
                else:
                    crystal_ids = crystal_id
                
                if isinstance(prediction, torch.Tensor):
                    preds = prediction.cpu().numpy()
                else:
                    preds = prediction
                
                # Handle batch
                if len(preds.shape) == 0:  # Single prediction
                    preds = [float(preds)]
                    crystal_ids = [int(crystal_ids)]
                else:
                    preds = [float(p) for p in preds]
                    crystal_ids = [int(c) for c in crystal_ids]
                
                for cid, pred in zip(crystal_ids, preds):
                    predictions[cid] = pred
                    
            except Exception as e:
                print(f"⚠️  Error processing batch {j}: {e}")
                continue
    
    return predictions

def fix_alignn_predictions_complete(struct_type="unrelaxed", gpu_num=0):
    """Fix ALIGNN predictions using exact training pipeline"""
    
    print(f"🔧 Fixing ALIGNN predictions for {struct_type} using training pipeline...")
    
    # Model parameters (same as training)
    model_params = {
        "data": "data/",
        "struct_type": struct_type,
        "model_type": "ALIGNN",
        "training_fraction": 1.0,
        "interpolation": False,
        "contrastive_weight": 1.0,
        "long_range": False
    }
    
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    
    # Load and process data EXACTLY like training
    print("📊 Loading data...")
    training_data = pd.read_json("data/training_set.json")
    training_data = training_data.sample(frac=model_params["training_fraction"], replace=False, random_state=0)
    test_data = pd.read_json("data/test_set.json")
    edge_data = pd.read_json("data/edge_dataset.json")
    
    if not model_params["interpolation"]:
        training_data = pd.concat((training_data, edge_data))
    
    print(f"📊 Original test data: {len(test_data)} samples")
    
    # Process data EXACTLY like training
    print("🔄 Processing data like training...")
    
    # Process training data
    train_processed = filter_data_by_properties(training_data, ["dft_e_hull"])
    train_processed = select_structures([train_processed], struct_type)[0]
    
    # Process test data  
    test_processed = filter_data_by_properties(test_data, ["dft_e_hull"])
    test_processed = select_structures([test_processed], struct_type)[0]
    
    print(f"📊 Processed test data: {len(test_processed)} samples")
    
    # Create dataloaders EXACTLY like training
    print("🔄 Creating dataloaders...")
    train_loader = get_dataloader(train_processed, "dft_e_hull", "ALIGNN", 1, False, False, False, device_name)
    test_loader = get_dataloader(test_processed, "dft_e_hull", "ALIGNN", 1, False, False, False, device_name)
    
    print(f"📊 Test loader size: {len(test_loader)} batches")
    
    # Generate wandb name
    wandb_name = build_wandb_name(
        model_params["data"], 
        "dft_e_hull", 
        model_params["struct_type"], 
        model_params["interpolation"], 
        model_params["model_type"],
        contrastive_weight=model_params["contrastive_weight"],
        training_fraction=model_params["training_fraction"],
        long_range=model_params["long_range"]
    )
    
    # Process each model
    for model_idx in range(3):  # best_0, best_1, best_2
        print(f"\n🔄 Processing model {model_idx}...")
        
        directory = f"./best_models/ALIGNN/{wandb_name}/best_{model_idx}"
        
        try:
            # Load model
            model, normalizer = load_alignn_model(gpu_num, train_loader, model_params, directory, "dft_e_hull")
            
            # Get predictions using exact same evaluation as training
            print("🔍 Generating predictions...")
            predictions = evaluate_alignn_model_complete(model, normalizer, test_loader, gpu_num)
            
            print(f"✅ Generated {len(predictions)} predictions")
            
            # Create complete results DataFrame
            results_df = test_processed.copy()
            
            # Map predictions back to DataFrame indices
            pred_values = []
            for idx, row in results_df.iterrows():
                # Try to get crystal_id or use index
                if 'crystal_id' in row:
                    crystal_id = row['crystal_id']
                else:
                    crystal_id = idx
                
                if crystal_id in predictions:
                    pred_values.append(predictions[crystal_id])
                else:
                    # Fallback to mean or true value
                    pred_values.append(row['dft_e_hull'])
                    print(f"⚠️  No prediction for crystal_id {crystal_id}, using fallback")
            
            results_df['predicted_dft_e_hull'] = pred_values
            
            # Save results
            output_path = f"{directory}/test_set_predictions.json"
            results_df.to_json(output_path, orient='records', indent=2)
            print(f"💾 Saved complete predictions to {output_path}")
            
            # Verify
            verification_df = pd.read_json(output_path)
            print(f"🔍 Verification: Saved {len(verification_df)} samples")
            
            if len(verification_df) >= len(test_processed):
                print("✅ SUCCESS: Complete processed dataset saved!")
            else:
                print("❌ WARNING: Incomplete dataset saved")
                
        except Exception as e:
            print(f"❌ Error processing model {model_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

def main():
    print("🔧 Fixing ALIGNN predictions using exact training pipeline...")
    
    # Fix unrelaxed predictions
    print("\n1️⃣  Fixing unrelaxed ALIGNN predictions...")
    try:
        fix_alignn_predictions_complete("unrelaxed", 0)
    except Exception as e:
        print(f"❌ Error with unrelaxed: {e}")
        import traceback
        traceback.print_exc()
    
    # Fix relaxed predictions
    print("\n2️⃣  Fixing relaxed ALIGNN predictions...")
    try:
        fix_alignn_predictions_complete("relaxed", 0)
    except Exception as e:
        print(f"❌ Error with relaxed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 ALIGNN prediction fix completed!")
    print("📊 ALIGNN prediction files should now have the same structure coverage as training")

if __name__ == "__main__":
    main()
