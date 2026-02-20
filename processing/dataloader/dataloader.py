import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch_geometric as tg
import json
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
from processing.dataloader.build_data import build_e3nn_data, construct_contrastive_dataset
from processing.dataloader.contrastive_data import CompDataLoader
from models.PerovskiteOrderingGCNNs_cgcnn.cgcnn.data import get_cgcnn_loader
import sys
sys.path.append('models/PerovskiteOrderingGCNNs_painn/')
from nff.data import Dataset, collate_dicts
from models.PerovskiteOrderingGCNNs_alignn.alignn.dataset import get_torch_dataset
from models.PerovskiteOrderingGCNNs_alignn.alignn.graphs import compute_bond_cosines
import torch
import dgl


class AlignnCollateFunction:
    """Picklable collate function for ALIGNN dataloader"""
    def __init__(self, device="cuda:0"):
        self.device = device
    
    def __call__(self, samples):
        """CPU-only collate function; device transfer is handled in training loop."""
        if len(samples) == 0:
            raise ValueError("Received empty sample list in ALIGNN collate function.")

        first = samples[0]
        if not isinstance(first, (list, tuple)):
            raise ValueError(f"Unexpected ALIGNN sample type: {type(first)}")

        line_graphs = None
        if len(first) == 5:
            # (graph, line_graph, lattice, label, id)
            graphs, line_graphs, lattices, labels, ids = map(list, zip(*samples))
        elif len(first) == 4:
            # Either:
            # (graph, line_graph, lattice, label) OR (graph, lattice, label, id)
            if isinstance(first[1], dgl.DGLGraph):
                graphs, line_graphs, lattices, labels = map(list, zip(*samples))
                ids = [None] * len(graphs)
            else:
                graphs, lattices, labels, ids = map(list, zip(*samples))
        elif len(first) == 3:
            # (graph, lattice, label)
            graphs, lattices, labels = map(list, zip(*samples))
            ids = [None] * len(graphs)
        else:
            raise ValueError(
                f"Unexpected ALIGNN sample length: {len(first)}. "
                "Expected 3, 4, or 5 fields."
            )
        
        # Build line graphs if dataset did not provide them.
        if line_graphs is None:
            line_graphs = []
            for graph in graphs:
                lg = graph.line_graph(shared=True)
                lg.apply_edges(compute_bond_cosines)
                line_graphs.append(lg)
        
        # Batch graphs on CPU first
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        
        # Convert lattices and labels to tensors on CPU first
        if isinstance(labels[0], torch.Tensor):
            labels_tensor = torch.stack(labels)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        if isinstance(lattices[0], torch.Tensor):
            lattices_tensor = torch.stack(lattices)
        else:
            lattices_tensor = torch.tensor(lattices, dtype=torch.float32)
        
        # Return in CGCNN format: (input_data, targets, crystal_ids)
        return (batched_graph, batched_line_graph, lattices_tensor), labels_tensor, ids


def get_dataloader(data, prop="dft_e_hull", model_type="CGCNN", batch_size=10, interpolation=True, per_site=False, long_range=False, device="cuda:0"):
    tqdm.pandas()
    pd.options.mode.chained_assignment = None # Disable the SettingWithCopy warning (due to pandas.apply as new column)
    
    data['ase_structure'] = data.progress_apply(lambda x: AseAtomsAdaptor.get_atoms(x['structure']), axis=1)
    data['idx'] = data.index

    if interpolation:
        prop += "_diff"

    if model_type == "CGCNN":
        data_loader = get_cgcnn_loader(data,prop,batch_size,per_site=per_site,long_range=long_range)
    elif model_type == "Painn":
        data_loader = get_painn_dataloader(data,prop,batch_size)
    elif model_type == "e3nn":
        data_loader = get_e3nn_dataloader(data,prop,batch_size,per_site=per_site)
    elif model_type == "e3nn_contrastive":
        data_loader = get_e3nn_contrastive_dataloader(data,prop,batch_size)
    elif model_type == "ALIGNN":
        data_loader = get_alignn_dataloader(data, prop, batch_size, device)
    else:
        raise ValueError("Model Type Not Supported")

    return data_loader


def get_painn_dataloader(data,prop,batch_size):

    data_props = dataframe_to_props_painn(data, prop)
    dataset = Dataset(data_props, units='eV', stack=True)
    f = open("processing/dataloader/atom_init.json")
    atom_inits = json.load(f)

    for key, value in atom_inits.items():
        atom_inits[key] = np.array(value, dtype=np.float32)

    dataset.generate_neighbor_list(cutoff=5.0, undirected=False)
    dataset.generate_atom_initializations(atom_inits)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_dicts, sampler=RandomSampler(dataset))

    return data_loader


def get_e3nn_dataloader(data,prop,batch_size,per_site):
    data['datapoint'] = data.progress_apply(lambda x: build_e3nn_data(x, prop, r_max=5.0,per_site=per_site), axis=1)
    data_loader = tg.loader.DataLoader(data['datapoint'].values, batch_size=batch_size, shuffle=True)

    return data_loader


def get_e3nn_contrastive_dataloader(data,prop,batch_size):
    comp_data = construct_contrastive_dataset(data,prop,r_max=5.0)
    data_loader = CompDataLoader(comp_data, batch_size=batch_size, shuffle=True)

    return data_loader


def get_alignn_dataloader(data, prop, batch_size, device="cuda:0"):
    """GPU-optimized ALIGNN dataloader with pre-computed graphs and tensor caching"""
    # Convert DataFrame to ALIGNN dataset and DataLoader
    # ALIGNN expects an 'atoms' column, so we need to convert ase_structure to atoms format
    
    # Create a copy of the data to avoid modifying the original
    alignn_data = data.copy()
    
    # Convert ase_structure to atoms format that ALIGNN expects
    def convert_ase_to_atoms_dict(ase_atoms):
        """Convert ASE Atoms object to JARVIS Atoms dictionary format expected by ALIGNN"""
        num_atoms = len(ase_atoms)
        return {
            'elements': list(ase_atoms.get_chemical_symbols()),
            'coords': ase_atoms.get_positions().tolist(),
            'lattice_mat': ase_atoms.get_cell().tolist(),
            'cartesian': True,
            'props': [''] * num_atoms  # Empty props for each atom
        }
    
    # Add atoms column by converting ase_structure
    alignn_data['atoms'] = alignn_data['ase_structure'].apply(convert_ase_to_atoms_dict)
    
    dataset = get_torch_dataset(
        dataset=alignn_data.to_dict("records"),
        target=prop,
        id_tag="idx",
        name="user_data",
        neighbor_strategy="k-nearest",
        atom_features="cgcnn",
        use_canonize=False,
        line_graph=True,
        cutoff=8.0,
        cutoff_extra=3.0,
        max_neighbors=12,
        classification=False,
        output_dir=".",
        tmp_name="alignn_tmp",
        dtype="float32",
    )
    
    # Disable multiprocessing to avoid CUDA tensor sharing issues
    # Use single worker and disable pin_memory to avoid multiprocessing problems
    num_workers = 0
    pin_memory = False
    persistent_workers = False
    print("Using single worker for ALIGNN dataloader to avoid CUDA multiprocessing issues")

    # Create picklable collate function
    collate_fn = AlignnCollateFunction(device)

    # Use custom GPU-optimized collate function
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    return data_loader


def dataframe_to_props_painn(df, target_prop):   
    prop_names = [target_prop]
    props = {}
    id_list = []
    nxyz_list = []
    props_list = {prop: [] for prop in prop_names}
    lattice_list = []    
    
    for index, row in df.iterrows():
        curr_struct = row['ase_structure']
        id_list.append(index)
        
        for prop in prop_names:
            props_list[prop].append(row[prop])
        
        n = np.asarray(curr_struct.numbers).reshape(-1,1)
        xyz = np.asarray(curr_struct.positions)
        curr_nxyz = np.concatenate((n, xyz), axis=1)
        nxyz_list.append(curr_nxyz)
        lattice_list.append(curr_struct.cell[:])
        
    props['crystal_id'] = id_list
    props['nxyz'] = nxyz_list
    
    for prop in prop_names:
        props[prop] = props_list[prop]
    
    props['lattice'] = lattice_list
        
    return props
