#!/usr/bin/env python
"""
Elastic Weight Consolidation (EWC) Module

This module implements functions for EWC in continual learning:
  - train_ewc: Performs training with EWC regularization.
  - estimate_fisher: Estimates and updates the Fisher Information for model parameters.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

def train_ewc(model, dataset, iters, lr, batch_size, device, current_task=None, ewc_lambda=100., task_classes=None):
    """
    Trains the model using Elastic Weight Consolidation (EWC) regularization.

    Args:
        model (torch.nn.Module): The neural network model.
        dataset (torch.utils.data.Dataset): Training dataset.
        iters (int): Number of iterations to train.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        device (torch.device): Device to run training on.
        current_task (int, optional): Current task index (EWC applied if > 0).
        ewc_lambda (float, optional): Regularization strength for EWC.
        task_classes (list): List of task-relevant class labels.

    Returns:
        list: Loss values for each iteration.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    data_loader_iter = iter(data_loader)
    
    # Map each task class to its index
    class_to_idx = {cls: idx for idx, cls in enumerate(task_classes)}
    
    for batch_index in range(iters):
        try:
            x, y = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            x, y = next(data_loader_iter)
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        
        # Select outputs corresponding to the current task classes
        current_indices = [class_to_idx[cls] for cls in task_classes]
        masked_outputs = outputs[:, current_indices]
        
        try:
            adjusted_labels = torch.tensor([class_to_idx[int(lbl.item())] for lbl in y.cpu()]).long().to(device)
        except KeyError as e:
            raise ValueError(f"Label {e.args[0]} not found in task_classes {task_classes}")
        
        loss = torch.nn.functional.cross_entropy(masked_outputs, adjusted_labels)
        
        # If using EWC (i.e., for tasks > 0), compute the EWC loss term.
        if current_task is not None and current_task > 0:
            ewc_losses = []
            for n, p in model.named_parameters():
                # Replace dots for buffer naming consistency.
                n_mod = n.replace('.', '__')
                mean = getattr(model, f'{n_mod}_EWC_param_values')
                fisher = getattr(model, f'{n_mod}_EWC_estimated_fisher')
                
                # Compute parameter difference, handling potential shape mismatches:
                if p.shape == mean.shape:
                    diff = p - mean
                elif p.dim() > 0 and p.shape[1:] == mean.shape[1:]:
                    min_dim = min(p.shape[0], mean.shape[0])
                    diff = p[:min_dim] - mean[:min_dim]
                else:
                    print(f"[INFO] Skipping EWC for parameter '{n}' due to shape mismatch: p {p.shape} vs mean {mean.shape}")
                    continue
                
                ewc_losses.append((fisher * (diff)**2).sum())
            
            ewc_loss = 0.5 * sum(ewc_losses)
            total_loss = loss + ewc_lambda * ewc_loss
        else:
            total_loss = loss
        
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
    
    return losses


def estimate_fisher(model, dataset, n_samples, device, ewc_gamma=1.0, batch_size=8):
    """
    Estimates the Fisher Information for model parameters and updates stored buffers.

    Args:
        model (torch.nn.Module): The neural network model.
        dataset (torch.utils.data.Dataset): Dataset used for Fisher estimation.
        n_samples (int): Number of samples to use for estimation.
        device (torch.device): Device to run computations on.
        ewc_gamma (float, optional): Factor for combining previous Fisher estimates.
        batch_size (int, optional): Batch size for Fisher estimation.

    Returns:
        None
    """
    # Initialize Fisher information tensor for each parameter.
    est_fisher_info = {n.replace('.', '__'): torch.zeros_like(p, device=device)
                       for n, p in model.named_parameters()}

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    n_samples = min(n_samples, len(dataset))
    sampled_indices = np.random.choice(len(dataset), size=n_samples, replace=False)
    subset_dataset = Subset(dataset, sampled_indices)
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
    
    total_processed = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        batch_size_actual = x.size(0)
        total_processed += batch_size_actual
        
        for n, p in model.named_parameters():
            param_name = n.replace('.', '__')
            if p.grad is not None:
                est_fisher_info[param_name] += (p.grad.detach() ** 2) * batch_size_actual

    # Average Fisher information over the processed samples.
    est_fisher_info = {n: p / total_processed for n, p in est_fisher_info.items()}

    # Update stored Fisher buffers, combining with previous estimates if available.
    for n, p in model.named_parameters():
        param_name = n.replace('.', '__')
        # Register current parameter values for future EWC computations.
        model.register_buffer(f'{param_name}_EWC_param_values', p.detach().clone())
        if hasattr(model, f'{param_name}_EWC_estimated_fisher'):
            prev_fisher = getattr(model, f'{param_name}_EWC_estimated_fisher')
            if p.shape == prev_fisher.shape:
                est_fisher_info[param_name] += ewc_gamma * prev_fisher
            elif p.dim() > 0 and p.shape[1:] == prev_fisher.shape[1:]:
                min_dim = min(p.shape[0], prev_fisher.shape[0])
                est_fisher_info[param_name][:min_dim] += ewc_gamma * prev_fisher[:min_dim]
            else:
                print(f"[INFO] Skipping EWC update for parameter '{n}' due to shape mismatch: current {p.shape} vs previous {prev_fisher.shape}")
        model.register_buffer(f'{param_name}_EWC_estimated_fisher', est_fisher_info[param_name])
    
    model.train()
