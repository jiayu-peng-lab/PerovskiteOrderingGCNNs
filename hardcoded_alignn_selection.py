#!/usr/bin/env python3
"""
Hard-coded ALIGNN model selection to find the best models
"""
import os
import shutil
import json
import pandas as pd

def find_best_alignn_models():
    """Find the best ALIGNN models from the hard-coded folders"""
    
    print("🔍 Finding best ALIGNN models from hard-coded folders...")
    
    # Hard-coded paths based on manual exploration
    unrelaxed_wandb_folder = "saved_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/wandb-i9iun9xf"
    relaxed_wandb_folder = "saved_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/wandb-kjahfk"
    
    print(f"📁 Unrelaxed ALIGNN folder: {unrelaxed_wandb_folder}")
    print(f"📁 Relaxed ALIGNN folder: {relaxed_wandb_folder}")
    
    # Process unrelaxed models
    print("\n🔄 Processing unrelaxed ALIGNN models...")
    unrelaxed_models = find_best_models_in_folder(unrelaxed_wandb_folder, "unrelaxed")
    
    # Process relaxed models
    print("\n🔄 Processing relaxed ALIGNN models...")
    relaxed_models = find_best_models_in_folder(relaxed_wandb_folder, "relaxed")
    
    # Copy best models to best_models directory
    print("\n📋 Copying best models to best_models directory...")
    copy_best_models(unrelaxed_models, "unrelaxed")
    copy_best_models(relaxed_models, "relaxed")
    
    print("\n🎉 ALIGNN model selection completed!")
    print("The best_models/ALIGNN directory now contains the truly best performing models.")

def find_best_models_in_folder(wandb_folder, struct_type):
    """Find the best models in a specific wandb folder"""
    
    if not os.path.exists(wandb_folder):
        print(f"❌ Folder does not exist: {wandb_folder}")
        return []
    
    # Get all observ directories
    observ_folders = [f for f in os.listdir(wandb_folder) if f.startswith('observ_')]
    print(f"📊 Found {len(observ_folders)} models in {wandb_folder}")
    
    # Check each model's training results
    model_results = []
    
    for folder in observ_folders:
        training_results_file = os.path.join(wandb_folder, folder, "training_results.json")
        
        if os.path.exists(training_results_file):
            try:
                with open(training_results_file, 'r') as f:
                    results = json.load(f)
                
                if 'validation_loss' in results:
                    validation_loss = results['validation_loss']
                    model_results.append({
                        'folder': folder,
                        'validation_loss': validation_loss,
                        'full_path': os.path.join(wandb_folder, folder)
                    })
                    print(f"  ✅ {folder}: validation_loss = {validation_loss:.6f}")
                else:
                    print(f"  ⚠️  {folder}: No validation_loss found")
            except Exception as e:
                print(f"  ❌ {folder}: Error reading results - {e}")
        else:
            print(f"  ❌ {folder}: No training_results.json found")
    
    # Sort by validation loss (lower is better)
    model_results.sort(key=lambda x: x['validation_loss'])
    
    print(f"\n🏆 Top 5 models by validation loss:")
    for i, model in enumerate(model_results[:5]):
        print(f"  {i+1}. {model['folder']}: {model['validation_loss']:.6f}")
    
    # Return top 3
    return model_results[:3]

def copy_best_models(best_models, struct_type):
    """Copy the best models to the best_models directory"""
    
    if not best_models:
        print(f"❌ No best models found for {struct_type}")
        return
    
    # Create destination directory
    dest_dir = f"best_models/ALIGNN/dft_e_hull_htvs_data_{struct_type}_ALIGNN"
    
    if os.path.exists(dest_dir):
        print(f"🗑️  Removing existing directory: {dest_dir}")
        shutil.rmtree(dest_dir)
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy each best model
    for i, model in enumerate(best_models):
        src_path = model['full_path']
        dest_path = os.path.join(dest_dir, f"best_{i}")
        
        print(f"📋 Copying {model['folder']} to best_{i}")
        shutil.copytree(src_path, dest_path)
        
        # Also copy the validation loss info
        with open(os.path.join(dest_path, "validation_loss.txt"), 'w') as f:
            f.write(f"Validation Loss: {model['validation_loss']:.6f}\n")
            f.write(f"Original Folder: {model['folder']}\n")
    
    print(f"✅ Copied {len(best_models)} best {struct_type} models to {dest_dir}")

if __name__ == "__main__":
    find_best_alignn_models()
