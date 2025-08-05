from models.PerovskiteOrderingGCNNs_alignn.alignn.models.alignn import ALIGNN, ALIGNNConfig
from models.PerovskiteOrderingGCNNs_cgcnn.cgcnn.model import Normalizer
from training.hyperparameters.default import get_default_alignn_hyperparameters
import numpy as np
from tqdm import tqdm
import torch

def get_alignn_model(hyperparameters, train_loader, per_site=False):
    """Create ALIGNN model following the same pattern as CGCNN and e3NN"""
    
    if hyperparameters == "default":
        hyperparameters = get_default_alignn_hyperparameters()
    
    # Extract ALIGNN-specific parameters from hyperparameters
    alignn_config = {
        'name': 'alignn',
        'embedding_features': hyperparameters.get('embedding_features', 64),
        'hidden_features': hyperparameters.get('hidden_features', 256),
        'alignn_layers': hyperparameters.get('alignn_layers', 4),
        'gcn_layers': hyperparameters.get('gcn_layers', 4),
        'output_features': 1,
        'classification': False
    }
    
    config = ALIGNNConfig(**alignn_config)
    model = ALIGNN(config)
    
    # Compute normalizer from training targets (same pattern as CGCNN and e3NN)
    training_labels = []
    for i, batch in enumerate(tqdm(train_loader)):
        # ALIGNN dataloader returns (graph, line_graph, lattice, target) when line_graph=True
        if isinstance(batch, (list, tuple)) and len(batch) >= 4:
            # Line graph format: (graph, line_graph, lattice, target)
            training_labels.append(batch[3].view(-1, 1))
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            # Regular format: (graph, target) or (graph, lattice, target)
            training_labels.append(batch[-1].view(-1, 1))
        elif hasattr(batch, 'target'):
            training_labels.append(batch.target.view(-1, 1))
        elif isinstance(batch, dict) and 'target' in batch:
            training_labels.append(batch['target'].view(-1, 1))
    
    # Concatenate all tensors and keep on GPU
    training_labels = torch.cat(training_labels, dim=0).squeeze()
    normalizer = Normalizer(training_labels)
    
    return model, normalizer 