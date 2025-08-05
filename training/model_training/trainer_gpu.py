import os
from models.PerovskiteOrderingGCNNs_painn.nff.train import Trainer, get_trainer, get_model, load_model, loss, hooks, metrics, evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from training.loss import contrastive_loss
from training.evaluate import evaluate_model
import torch
from torch.autograd import Variable
from tqdm import tqdm
import time
import dgl


def trainer_gpu(model, normalizer, model_type, train_loader, val_loader, hyperparameters, OUTDIR, gpu_num, train_eval_loader=None, contrastive_weight=1.0):
    """GPU-optimized trainer that ensures proper GPU usage for DGL"""
    
    hyperparameters["MaxEpochs"] = 100
    
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    if "contrastive" in model_type:
        loss_fn = contrastive_loss 
    else:
        loss_fn = torch.nn.L1Loss()
    
    if model_type == "Painn":
        best_model = train_painn(model, train_loader, val_loader, hyperparameters, OUTDIR, gpu_num)
    elif model_type == "ALIGNN":
        best_model = train_alignn_gpu(model, normalizer, train_loader, val_loader, hyperparameters, OUTDIR, gpu_num)
    else:
        best_model = train_CGCNN_e3nn(model, normalizer, model_type, loss_fn, contrastive_loss, train_loader, val_loader, hyperparameters, OUTDIR, gpu_num, train_eval_loader, contrastive_weight)

    return best_model, loss_fn


def train_alignn_gpu(model, normalizer, train_loader, val_loader, hyperparameters, OUTDIR, gpu_num):
    """GPU-optimized training function specifically for ALIGNN model with DGL"""
    
    # Set up device
    device_name = f"cuda:{gpu_num}"
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    print(f"Using CUDA device: {device}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    best_validation_error = 99999999
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10**hyperparameters["log_lr"])
    max_epochs = hyperparameters['MaxEpochs']
    scheduler = ReduceLROnPlateau(
            optimizer,
            patience=hyperparameters["reduceLR_patience"],
            factor=0.5, 
            min_lr=1e-7, 
        )

    results = {}
    history = []
    loss_fn = torch.nn.L1Loss()
    
    for epoch in range(max_epochs):
        model.train()
        start_time = time.time()
        
        for j, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # ALIGNN returns (graph, line_graph, lattice, label) when line_graph=True
            graph, line_graph, lattice, target = batch
            
            # Move to device - ensure DGL graphs are on GPU
            try:
                # Move DGL graphs to GPU
                graph = graph.to(device)
                line_graph = line_graph.to(device)
                
                # Move tensors to GPU
                if isinstance(lattice, torch.Tensor):
                    lattice = lattice.to(device)
                else:
                    lattice = torch.tensor(lattice, dtype=torch.float32, device=device)
                
                if isinstance(target, torch.Tensor):
                    target = target.to(device)
                else:
                    target = torch.tensor(target, dtype=torch.float32, device=device)
                    
            except Exception as e:
                print(f"Error moving data to GPU: {e}")
                print("Falling back to CPU")
                device = torch.device("cpu")
                model = model.to(device)
                graph = graph.to(device)
                line_graph = line_graph.to(device)
                if isinstance(lattice, torch.Tensor):
                    lattice = lattice.to(device)
                else:
                    lattice = torch.tensor(lattice, dtype=torch.float32, device=device)
                if isinstance(target, torch.Tensor):
                    target = target.to(device)
                else:
                    target = torch.tensor(target, dtype=torch.float32, device=device)
            
            # Forward pass - ALIGNN expects (g, lg, lat) tuple
            output = model((graph, line_graph, lattice)).view(-1)
            
            # Loss calculation
            loss = loss_fn(normalizer.denorm(output).view(target.shape), target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clear GPU cache periodically
            if j % 10 == 0:
                torch.cuda.empty_cache()

        end_time = time.time()
        wall = end_time - start_time    
        
        print(f"Epoch {epoch+1}/{max_epochs}, Time: {wall:.2f}s, GPU Memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
        model.eval()
        
        # Evaluation
        predictions, targets, train_avg_loss = evaluate_model(model, normalizer, "ALIGNN", train_loader, loss_fn, gpu_num)
        predictions, targets, valid_avg_loss = evaluate_model(model, normalizer, "ALIGNN", val_loader, loss_fn, gpu_num)
        
        results = record_keep(history, results, epoch, wall, optimizer, valid_avg_loss, train_avg_loss, model, "standard")
        validation_loss = valid_avg_loss[0]

        if (epoch == 0) or (validation_loss < best_validation_error):
            best_validation_error = validation_loss
            with open(OUTDIR + '/best_model.torch', 'wb') as f:
                torch.save(results, f)

        if scheduler is not None:
            scheduler.step(validation_loss)

    with open(OUTDIR + '/final_model.torch', 'wb') as f:
        torch.save(results, f)

    model_state = torch.load(OUTDIR + '/best_model.torch', map_location=torch.device('cpu'))['state']
    model.load_state_dict(model_state)
    model.to(device)
    return model


def train_painn(model, train_loader, val_loader, hyperparameters, OUTDIR, gpu_num):
    """Training function for Painn model"""
    prop_names = model.output_keys
    loss_fn = loss.build_mae_loss(loss_coef = {prop: 1.0 for prop in prop_names})
    train_metrics = [metrics.MeanAbsoluteError(prop) for prop in prop_names]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=10**hyperparameters["log_lr"])
    num_epochs = hyperparameters["MaxEpochs"]
    
    train_hooks = [
        hooks.MaxEpochHook(num_epochs),
        hooks.CSVHook(
            OUTDIR, 
            metrics=train_metrics
        ), 
        hooks.PrintingHook(
            OUTDIR, 
            metrics=train_metrics, 
            separator = ' | ', 
            time_strf='%M:%S'
        ), 
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer, 
            patience=hyperparameters["reduceLR_patience"], 
            factor=0.5, 
            min_lr=1e-7, 
            window_length=1, 
            stop_after_min=True)
    ]
    
    T = Trainer(
        model_path=OUTDIR, 
        model=model, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=train_hooks,
        mini_batches=1
    )
    
    T.train(device=gpu_num, n_epochs=num_epochs)

    return T.get_best_model()


def train_CGCNN_e3nn(model, normalizer, model_type, loss_fn, contrastive_loss_fn, train_loader, val_loader, hyperparameters, OUTDIR, gpu_num, train_eval_loader, contrastive_weight):
    """Training function for CGCNN and e3nn models"""
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    torch.cuda.set_device(device)

    best_validation_error = 99999999
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10**hyperparameters["log_lr"])
    max_epochs = hyperparameters['MaxEpochs']
    scheduler = ReduceLROnPlateau(
            optimizer,
            patience=hyperparameters["reduceLR_patience"],
            factor=0.5, 
            min_lr=1e-7, 
        )

    results = {}
    history = []
    
    for epoch in range(max_epochs):
        model.train()
        start_time = time.time()
        
        for j, d in enumerate(train_loader):
            if model_type == "CGCNN":
                input_struct = d[0]
                target = d[1]
                input_var = (Variable(input_struct[0].cuda(non_blocking=True)),
                             Variable(input_struct[1].cuda(non_blocking=True)),
                             input_struct[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input_struct[3]])
                output = model(*input_var).view(-1)
                target = Variable(target.cuda(non_blocking=True))
            else:
                d.to(device)
                output = model(d)
                target = d.target
                
            prediction = normalizer.denorm(output)
            
            if "contrastive" in model_type:
                loss, direct_loss, contrastive_loss = loss_fn(prediction, target, d.comp, contrastive_weight)
            else:
                loss = loss_fn(prediction, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        wall = end_time - start_time    
        
        model.eval()
        
        # Evaluation
        predictions, targets, train_avg_loss = evaluate_model(model, normalizer, model_type, train_loader, loss_fn, gpu_num)
        predictions, targets, valid_avg_loss = evaluate_model(model, normalizer, model_type, val_loader, loss_fn, gpu_num)
        
        results = record_keep(history, results, epoch, wall, optimizer, valid_avg_loss, train_avg_loss, model, "standard")
        validation_loss = valid_avg_loss[0]

        if (epoch == 0) or (validation_loss < best_validation_error):
            best_validation_error = validation_loss
            with open(OUTDIR + '/best_model.torch', 'wb') as f:
                torch.save(results, f)

        if scheduler is not None:
            scheduler.step(validation_loss)

    with open(OUTDIR + '/final_model.torch', 'wb') as f:
        torch.save(results, f)

    model_state = torch.load(OUTDIR + '/best_model.torch', map_location=torch.device('cpu'))['state']
    model.load_state_dict(model_state)
    model.to(device)
    return model


def record_keep(history, results, epoch, wall, optimizer, valid_avg_loss, train_avg_loss, model, eval_type, contrastive_weight=None):
    """Record training history and results"""
    history.append({
        'epoch': epoch,
        'wall': wall,
        'lr': optimizer.param_groups[0]['lr'],
        'train_loss': train_avg_loss[0],
        'val_loss': valid_avg_loss[0]
    })
    
    if contrastive_weight is not None and len(valid_avg_loss) > 2:
        history[-1]['train_direct_loss'] = train_avg_loss[1]
        history[-1]['train_contrastive_loss'] = train_avg_loss[2]
        history[-1]['val_direct_loss'] = valid_avg_loss[1]
        history[-1]['val_contrastive_loss'] = valid_avg_loss[2]
    
    results = {
        'history': history,
        'state': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    
    return results 