#!/usr/bin/env python
"""
Diagnostic: Are gradients flowing to the SAT head?
This runs one training step and checks if the SAT head gradients are non-zero.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from forge.embeddings import Forge
from forge.processor import SATProcessor
from forge.utils import Constants, load_pickle

if __name__ == "__main__":
    
    print(f"\n{'='*80}")
    print(f"GRADIENT FLOW DIAGNOSTIC")
    print(f"{'='*80}\n")
    
    # Load a pretrained model
    forge = Forge('../forge/configs/old_train_config.yaml')
    forge.load_model(input_forge_pkl='../models/iclr_forge_pretrain_trained.pkl',
                     model_type=Constants.FORGE_FINE_TUNE_SAT,
                     strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forge.to(device)
    forge.train()
    
    # Load sat_to_satinfo
    sat_to_satinfo = load_pickle('../models/g4satbench_easy_sr_train_sat_to_satinfo.pkl')
    sat_files = list(sat_to_satinfo.keys())[:10]  # Use just first 10
    
    print(f"Testing gradient flow on {len(sat_files)} SAT instances...\n")
    
    # Create SAT head if not present
    if not hasattr(forge, 'sat_satisfiability_layer'):
        forge.has_sat_satisfiability_head = True
        forge.sat_satisfiability_layer = nn.Linear(forge.updated_input_dim, 1).to(device)
        nn.init.xavier_uniform_(forge.sat_satisfiability_layer.weight)
        if forge.sat_satisfiability_layer.bias is not None:
            nn.init.zeros_(forge.sat_satisfiability_layer.bias)
    
    # Create optimizer
    optimizer = torch.optim.Adam(forge.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Start Gurobi
    import gurobipy as gp
    gurobi_env = gp.Env()
    
    # Run one training step
    gurobi_model = gp.read(sat_files[0], env=gurobi_env)
    satinfo = SATProcessor._sat_model_to_satinfo(gurobi_model)
    
    if satinfo is None:
        print("ERROR: Could not convert SAT file to SATInfo")
        sys.exit(1)
    
    # Move tensors to device
    feature_tensor = satinfo.feature_tensor.to(device)
    edge_index = satinfo.edge_index.to(device)
    edge_weight = satinfo.edge_weight.to(device) if satinfo.edge_weight is not None else None
    
    # Compute embeddings
    h_list, logits, loss, indices, codebook_ = forge.forward(
        feature_tensor,
        satinfo.num_clauses,
        satinfo.num_vars,
        edge_index,
        edge_weight
    )
    
    # Get SAT head prediction
    # h_list[-1] has shape [num_nodes, 1], average across nodes
    sat_pred_logit = torch.mean(h_list[-1])
    
    # Apply sigmoid to get probability
    sat_pred_prob = torch.sigmoid(sat_pred_logit)
    
    # Get label from filename (True if SAT, False if UNSAT)
    from pathlib import Path
    filename = Path(sat_files[0]).name
    sat_label = "unsat" not in filename.lower()  # True if SAT, False if UNSAT
    sat_label_tensor = torch.tensor(float(sat_label), dtype=torch.float32, device=device)
    
    print(f"Instance: {filename}")
    print(f"True label (from filename): {sat_label}")
    print(f"Model prediction (logit): {sat_pred_logit.item():.4f}")
    print(f"Model prediction (prob):  {sat_pred_prob.item():.4f}")
    
    # Compute loss using BCE with logits
    loss_bce = nn.BCEWithLogitsLoss()(sat_pred_logit.unsqueeze(0), sat_label_tensor.unsqueeze(0))
    
    print(f"Loss: {loss_bce.item():.6f}\n")
    
    # Backprop
    optimizer.zero_grad()
    loss_bce.backward()
    
    print(f"{'='*80}")
    print(f"GRADIENT CHECK - Are gradients non-zero?")
    print(f"{'='*80}\n")
    
    # Check SAT head gradients
    sat_head_weight_grad = forge.sat_satisfiability_layer.weight.grad
    sat_head_bias_grad = forge.sat_satisfiability_layer.bias.grad if forge.sat_satisfiability_layer.bias is not None else None
    
    print(f"SAT head weight gradient norm: {sat_head_weight_grad.norm().item():.6f}")
    if sat_head_bias_grad is not None:
        print(f"SAT head bias gradient norm:   {sat_head_bias_grad.norm().item():.6f}")
    
    # Check if gradients exist for encoder
    graph_layer_1_grad = None
    for param in forge.graph_layer_1.parameters():
        if param.grad is not None:
            graph_layer_1_grad = param.grad.norm().item()
            break
    
    if graph_layer_1_grad is not None:
        print(f"graph_layer_1 gradient norm:   {graph_layer_1_grad:.6f}")
    else:
        print(f"graph_layer_1 gradient norm:   ZERO (no gradients)")
    
    print(f"\n{'='*80}\n")
    
    # Check if SAT head gradients are zero
    if sat_head_weight_grad.norm().item() == 0:
        print(f"⚠️  WARNING: SAT head has ZERO gradients!")
        print(f"This means gradients are not flowing through the model.\n")
    else:
        print(f"✓ SAT head gradients are non-zero (gradients are flowing)\n")
    
    gurobi_model.dispose()
    gurobi_env.dispose()
