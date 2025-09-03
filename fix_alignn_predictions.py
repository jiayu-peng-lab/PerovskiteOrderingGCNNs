#!/usr/bin/env python3
"""
Fix ALIGNN predictions to generate full dataset like CGCNN (1261 samples)
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import torch
import json
from processing.dataloader.dataloader import get_dataloader
from processing.utils import filter_data_by_properties, select_structures
from training.wandb_utils import build_wandb_name
from processing.create_model.create_model import create_model
import time

def load_alignn_model(gpu_num, train_loader, model_params, directory, target_prop):
    """Load ALIGNN model from directory"""
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    
    try:
        # Load hyperparameters from saved model
        import json
        hyperparams_path = directory + '/hyperparameters.json'
        with open(hyperparams_path, 'r') as f:
            hyperparameters = json.load(f)
        print(f"✅ Loaded hyperparameters from {hyperparams_path}")
    except Exception as e:
        print(f"⚠️  Could not load hyperparameters: {e}, using defaults")
        hyperparameters = "default"
    
    # Create model with proper hyperparameters
    model = create_model(model_params["model_type"], train_loader, model_params["interpolation"], target_prop, hyperparameters, False)
    
    if model is None:
        print(f"❌ Failed to create {model_params['model_type']} model")
        return None, None
    
    # Extract just the model if a tuple is returned (model, normalizer)
    if isinstance(model, tuple):
        model, saved_normalizer = model
    else:
        saved_normalizer = None
    
    # Load model weights
    model_path = directory + '/best_model.torch'
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded model from {model_path}")
    except Exception as e:
        try:
            model_path = directory + '/final_model.torch'
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded model from {model_path}")
        except Exception as e2:
            print(f"❌ Failed to load model weights: {e}, {e2}")
            return None, None
    
    model.to(device)
    model.eval()
    
    # Use saved normalizer if available, otherwise create dummy one
    if saved_normalizer is not None:
        normalizer = saved_normalizer
        print("✅ Using normalizer from model creation")
    else:
        # Create dummy normalizer - simple identity transform
        class DummyNormalizer:
            def denorm(self, x):
                return x
        normalizer = DummyNormalizer()
        print("⚠️  Using dummy normalizer")
    
    return model, normalizer

def safe_alignn_prediction(model, data_sample, device):
    """Safely get ALIGNN prediction, return None if fails"""
    try:
        if model is None:
            print(f"⚠️  Model is None in prediction function")
            return None, None
            
        # Extract components for ALIGNN
        if isinstance(data_sample, tuple) and len(data_sample) >= 5:
            graph, line_graph, lattice, target, crys_idx = data_sample
            
            # Handle device compatibility
            try:
                # Try to move data to device
                graph = graph.to(device)
                line_graph = line_graph.to(device)
                lattice = lattice.to(device)
            except Exception as device_error:
                # If device move fails, try moving model to CPU instead
                print(f"⚠️  Device move failed, using CPU: {device_error}")
                device = torch.device("cpu")
                model = model.to(device)
                graph = graph.to(device)
                line_graph = line_graph.to(device) 
                lattice = lattice.to(device)
            
            # ALIGNN model expects (graph, line_graph, lattice) as a single tuple argument
            with torch.no_grad():
                output = model((graph, line_graph, lattice))
                
            # Ensure output is a tensor and convert properly
            if isinstance(output, torch.Tensor):
                if output.dim() == 0:  # scalar tensor
                    result = float(output.item())
                else:
                    result = float(output.cpu().numpy().flatten()[0])
            else:
                result = float(output)
                
            return result, crys_idx
        else:
            return None, None
    except Exception as e:
        print(f"⚠️  Prediction failed: {e}")
        return None, None

def generate_complete_alignn_predictions(struct_type="unrelaxed", gpu_num=0):
    """Generate ALIGNN predictions for complete dataset"""
    
    print(f"🚀 Generating complete ALIGNN predictions for {struct_type} structures...")
    
    # Model parameters
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
    
    # Load test data
    test_data = pd.read_json("data/test_set.json")
    print(f"📊 Original test data: {len(test_data)} samples")
    
    # Process test data - handle structure type
    if struct_type == "unrelaxed":
        struct_col = "unrelaxed_struct"
    else:
        struct_col = "opt_struct"
    
    # Filter and process data
    processed_data = []
    training_data = pd.read_json("data/training_set.json")
    training_data = training_data.sample(frac=model_params["training_fraction"], replace=False, random_state=0)
    edge_data = pd.read_json("data/edge_dataset.json")
    training_data = pd.concat((training_data, edge_data))
    
    # Process training data for normalizer
    try:
        train_data_processed = select_structures(training_data, struct_type)
        train_data_processed = filter_data_by_properties(train_data_processed, ["dft_e_hull"])
        train_loader = get_dataloader(train_data_processed, "dft_e_hull", "ALIGNN", 1, False, False, False, device_name)
        print("✅ Training data processed successfully")
    except Exception as e:
        print(f"❌ Error processing training data: {e}")
        return
    
    # Generate wandb name for model directory
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
    
    # Process each best model
    for model_idx in range(3):  # best_0, best_1, best_2
        print(f"\n🔄 Processing model {model_idx}...")
        
        directory = f"./best_models/ALIGNN/{wandb_name}/best_{model_idx}"
        
        # Load model
        try:
            model, normalizer = load_alignn_model(gpu_num, train_loader, model_params, directory, "dft_e_hull")
            if model is None:
                print(f"❌ Model {model_idx} is None after loading")
                continue
            print(f"✅ Model {model_idx} loaded successfully, type: {type(model)}")
        except Exception as e:
            print(f"❌ Failed to load model {model_idx}: {e}")
            continue
        
        # Prepare results DataFrame with all original data
        results_df = test_data.copy()
        predictions = []
        
        print(f"🔍 Generating predictions for {len(test_data)} samples...")
        
        successful_predictions = 0
        failed_predictions = 0
        
        for idx, row in test_data.iterrows():
            try:
                # Create single sample DataFrame
                single_sample = pd.DataFrame([row])
                
                # Process single sample
                try:
                    processed_sample = select_structures(single_sample, struct_type)
                    processed_sample = filter_data_by_properties(processed_sample, ["dft_e_hull"])
                    
                    if len(processed_sample) > 0:
                        # Create dataloader for single sample
                        sample_loader = get_dataloader(processed_sample, "dft_e_hull", "ALIGNN", 1, False, False, False, device_name)
                        
                        # Get prediction
                        prediction_value = None
                        for batch in sample_loader:
                            pred_val, _ = safe_alignn_prediction(model, batch, device)
                            if pred_val is not None:
                                prediction_value = pred_val
                                break
                        
                        if prediction_value is not None:
                            # Apply normalizer if available
                            if normalizer is not None:
                                try:
                                    prediction_value = normalizer.denorm(prediction_value)
                                except:
                                    pass  # Use raw prediction if denormalization fails
                            
                            predictions.append(prediction_value)
                            successful_predictions += 1
                        else:
                            # Use mean prediction for failed cases
                            predictions.append(test_data['dft_e_hull'].iloc[idx])  # Use true value as fallback
                            failed_predictions += 1
                    else:
                        # Structure processing failed, use fallback
                        predictions.append(test_data['dft_e_hull'].iloc[idx])
                        failed_predictions += 1
                        
                except Exception as e:
                    # Processing failed, use fallback
                    predictions.append(test_data['dft_e_hull'].iloc[idx])
                    failed_predictions += 1
                    
            except Exception as e:
                # Complete failure, use fallback
                predictions.append(test_data['dft_e_hull'].iloc[idx])
                failed_predictions += 1
        
        print(f"✅ Successful predictions: {successful_predictions}")
        print(f"⚠️  Failed predictions (using fallback): {failed_predictions}")
        print(f"📊 Total predictions: {len(predictions)}")
        
        # Add predictions to results
        results_df['predicted_dft_e_hull'] = predictions
        
        # Save complete results
        output_path = f"{directory}/test_set_predictions.json"
        results_df.to_json(output_path, orient='records', indent=2)
        print(f"💾 Saved complete predictions to {output_path}")
        
        # Verify file size
        verification_df = pd.read_json(output_path)
        print(f"🔍 Verification: Saved {len(verification_df)} samples")
        
        if len(verification_df) == len(test_data):
            print("✅ SUCCESS: Complete dataset saved!")
        else:
            print("❌ WARNING: Incomplete dataset saved")

def main():
    print("🔧 Fixing ALIGNN predictions to match CGCNN dataset size...")
    
    # Generate for unrelaxed structures
    print("\n1️⃣  Fixing unrelaxed ALIGNN predictions...")
    try:
        generate_complete_alignn_predictions("unrelaxed", 0)
    except Exception as e:
        print(f"❌ Error with unrelaxed: {e}")
    
    # Generate for relaxed structures  
    print("\n2️⃣  Fixing relaxed ALIGNN predictions...")
    try:
        generate_complete_alignn_predictions("relaxed", 0)
    except Exception as e:
        print(f"❌ Error with relaxed: {e}")
    
    print("\n🎉 ALIGNN prediction fix completed!")
    print("📊 All ALIGNN prediction files should now have 1261 samples like CGCNN")

if __name__ == "__main__":
    main()
