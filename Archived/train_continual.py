#!/usr/bin/env python
"""
Continual Learning Training Pipeline

This script:
  - Loads a YAML configuration file.
  - Runs a continual learning pipeline with per-task hyperparameter optimization (HPO).
  - For each task:
      * Tunes hyperparameters on task-specific training data.
      * Retrains the model on the entire task using the best hyperparameters.
      * Saves a checkpoint.
      * Updates the model’s output layer (before training starts on the task) to accommodate new classes.
      * Evaluates the model on:
            - The current task’s test dataset (filtered to only current task classes).
            - The cumulative test dataset (all tasks up to the current one, using global mapping).
      * Logs training/validation curves, GPU memory usage, and confusion matrices to TensorBoard.
  - Returns the final trained model.
"""

import os
import time
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.exceptions import TrialPruned
import matplotlib.pyplot as plt

# Import shared utilities and dataset class.
from utils import train_one_epoch, validate_one_epoch, evaluate_model, save_model_checkpoint, update_model_output, apply_mask
from Wafer_data_dataset_resize import WaferMapDataset

# ------------------------------------------------------------------------------
# Helper function to print model information.
def print_model_info(model, task_idx):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    output_units = None
    if hasattr(model, 'fc'):
        output_units = model.fc.out_features
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, torch.nn.Sequential):
            last_layer = list(model.classifier.children())[-1]
            if hasattr(last_layer, 'out_features'):
                output_units = last_layer.out_features
    print(f"[MODEL INFO] Task {task_idx}: {model.__class__.__name__} has total {total_params} parameters ({trainable_params} trainable).")
    if output_units is not None:
        print(f"[MODEL INFO] Task {task_idx}: Output layer has {output_units} units.")

# ------------------------------------------------------------------------------
def train_ewc(model, dataset, iters, lr, batch_size, device, current_task=None, ewc_lambda=100., task_classes=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    data_loader_iter = iter(data_loader)
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
        current_indices = [class_to_idx[cls] for cls in task_classes]
        masked_outputs = outputs[:, current_indices]

        try:
            adjusted_labels = torch.tensor([class_to_idx[int(lbl.item())] for lbl in y.cpu()]).long().to(device)
        except KeyError as e:
            raise ValueError(f"Label {e.args[0]} not found in task_classes {task_classes}")

        loss = torch.nn.functional.cross_entropy(masked_outputs, adjusted_labels)

        if current_task is not None and current_task > 0:
            ewc_losses = []
            for n, p in model.named_parameters():
                n = n.replace('.', '__')
                mean = getattr(model, f'{n}_EWC_param_values')
                fisher = getattr(model, f'{n}_EWC_estimated_fisher')
                ewc_losses.append((fisher * (p - mean)**2).sum())
            ewc_loss = 0.5 * sum(ewc_losses)
            total_loss = loss + ewc_lambda * ewc_loss
        else:
            total_loss = loss

        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    return losses

def estimate_fisher(model, dataset, n_samples, device, ewc_gamma=1.0):
    est_fisher_info = {}
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        est_fisher_info[n] = p.detach().clone().zero_()
    
    mode = model.training
    model.eval()
    data_loader = DataLoader(dataset, batch_size=1)

    for index, (x, y) in enumerate(data_loader):
        if n_samples is not None and index >= n_samples:
            break
        x = x.to(device)
        output = model(x)
        with torch.no_grad():
            label_weights = torch.nn.functional.softmax(output, dim=1)
        
        for label_index in range(output.shape[1]):
            label = torch.LongTensor([label_index]).to(device)
            negloglikelihood = torch.nn.functional.cross_entropy(output, label)
            model.zero_grad()
            negloglikelihood.backward(retain_graph=(label_index+1 < output.shape[1]))
            for n, p in model.named_parameters():
                n = n.replace('.', '__')
                if p.grad is not None:
                    est_fisher_info[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)
    
    est_fisher_info = {n: p / (index + 1) for n, p in est_fisher_info.items()}
    
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        model.register_buffer(f'{n}_EWC_param_values', p.detach().clone())
        if hasattr(model, f'{n}_EWC_estimated_fisher'):
            existing_values = getattr(model, f'{n}_EWC_estimated_fisher')
            est_fisher_info[n] += ewc_gamma * existing_values
        model.register_buffer(f'{n}_EWC_estimated_fisher', est_fisher_info[n])
    
    model.train(mode=mode)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["dataset"]["path"] = os.path.expanduser(config["dataset"]["path"])
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning Training Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    return parser.parse_args()

def tune_hyperparameters_for_task(config, task_dataset, model_factory, device, num_epochs, num_trials, num_classes):
    """
    Performs HPO (via Optuna) on the provided task dataset and returns the best lr and weight decay.
    Evaluates based on the average validation loss.
    """
    from sklearn.model_selection import StratifiedKFold

    # Helper function to get labels from a dataset or Subset.
    def get_labels(ds):
        if hasattr(ds, 'y'):
            return np.array(ds.y)
        elif isinstance(ds, Subset):
            return np.array(ds.dataset.y)[ds.indices]
        else:
            raise AttributeError("Dataset does not have attribute 'y'")

    def objective(trial):
        trial_lr = trial.suggest_float('lr',
                                       float(config["experiment"]["suggest"]["lr"]["low"]),
                                       float(config["experiment"]["suggest"]["lr"]["high"]),
                                       log=config["experiment"]["suggest"]["lr"]["log"])
        trial_weight_decay = trial.suggest_float('weight_decay',
                                                 float(config["experiment"]["suggest"]["weight_decay"]["low"]),
                                                 float(config["experiment"]["suggest"]["weight_decay"]["high"]),
                                                 log=config["experiment"]["suggest"]["weight_decay"]["log"])
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        all_indices = np.arange(len(task_dataset))
        all_labels = get_labels(task_dataset)
        fold_losses = []
        for train_idx, val_idx in skf.split(all_indices, all_labels):
            train_subset = Subset(task_dataset, train_idx)
            val_subset = Subset(task_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
            model, criterion, optimizer = model_factory(trial_lr, trial_weight_decay, num_classes, device)
            best_val_loss = float('inf')
            for epoch in range(config["experiment"]["num_epochs"]):
                t_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
                v_loss, _ = validate_one_epoch(model, val_loader, criterion, device)
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
            fold_losses.append(best_val_loss)
        return np.mean(fold_losses)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)
    best_trial = study.best_trial
    return best_trial.params['lr'], best_trial.params['weight_decay']

def evaluate_current_and_cumulative(config, model, device, task_list, current_task_index):
    """
    Evaluate the model on:
      (a) The current task's test set (filtered to only current task classes).
      (b) The cumulative test set from tasks 0 to current_task_index (global mapping).
    Logs the results to TensorBoard and prints them.
    """
    from Wafer_data_dataset_resize import WaferMapDataset
    from sklearn.metrics import classification_report
    # Build global mapping up to current task.
    cumulative_classes = [cls for sublist in task_list[:current_task_index+1] for cls in sublist]
    current_task_classes = task_list[current_task_index]
    
    # Create the global test dataset with the global mapping.
    global_test_ds = WaferMapDataset(
        file_path=config["dataset"]["path"],
        split="test",
        oversample=False,
        target_dim=(224, 224),
        task_classes=cumulative_classes
    )
    # Filter indices for current task only.
    current_indices = np.where(np.isin(np.array(global_test_ds.y), current_task_classes))[0]
    current_test_ds = Subset(global_test_ds, current_indices)
    
    current_loader = DataLoader(current_test_ds, batch_size=64, shuffle=False)
    preds, labels = evaluate_model(model, current_loader, device)
    current_acc = (preds == labels).mean()
    current_report = classification_report(labels, preds, digits=3)
    print(f"[Evaluation] Task {current_task_index} test set accuracy (current classes): {current_acc:.4f}")
    print(f"[Evaluation] Classification Report for Task {current_task_index} (current classes):\n{current_report}")
    
    # Cumulative evaluation: use the full global test dataset.
    cumu_loader = DataLoader(global_test_ds, batch_size=64, shuffle=False)
    cumu_preds, cumu_labels = evaluate_model(model, cumu_loader, device)
    cumu_acc = (cumu_preds == cumu_labels).mean()
    cumu_report = classification_report(cumu_labels, cumu_preds, digits=3)
    print(f"[Evaluation] Cumulative test set accuracy (tasks 0 to {current_task_index}): {cumu_acc:.4f}")
    print(f"[Evaluation] Cumulative Classification Report (tasks 0 to {current_task_index}):\n{cumu_report}")
    
    # Log results to TensorBoard.
    eval_log_dir = os.path.join("Logs", "evaluation", f"task{current_task_index}")
    writer = SummaryWriter(log_dir=eval_log_dir)
    writer.add_scalar("Test/Current_Accuracy", current_acc, global_step=current_task_index)
    writer.add_text("Test/Current_Classification_Report", current_report, global_step=current_task_index)
    writer.add_scalar("Test/Cumulative_Accuracy", cumu_acc, global_step=current_task_index)
    writer.add_text("Test/Cumulative_Classification_Report", cumu_report, global_step=current_task_index)
    writer.close()

def continual_training_pipeline(config, model_factory, device):
    ts = config["experiment"].get("timestamp", "default")
    base_log_dir = config["experiment"].get("log_dir", "logs")
    method = config["experiment"].get("continual_method", "ewc")
    print(f"[INFO] Starting continual learning training pipeline with method: {method}")
    print("[INFO] Configuration being used:")
    print(yaml.dump(config, default_flow_style=False))
    task_list = config["experiment"]["task_list"]  # e.g., [[0,1], [2,3], [4,5]]

    model = None

    for task_idx, local_classes in enumerate(task_list):
        # Global (cumulative) classes: union of classes from task 0 to current.
        global_classes = [cls for sublist in task_list[:task_idx+1] for cls in sublist]
        global_num_classes = len(global_classes)
        print(f"\n[INFO] Starting Task {task_idx} with:")
        print(f"        Current task classes: {local_classes}")
        print(f"        Global classes so far: {global_classes}")

        # Create global training dataset and then filter for current task.
        global_train_ds = WaferMapDataset(
            file_path=config["dataset"]["path"],
            split="train",
            oversample=False,
            target_dim=(224, 224),
            task_classes=global_classes
        )
        train_indices = np.where(np.isin(np.array(global_train_ds.y), local_classes))[0]
        train_dataset = Subset(global_train_ds, train_indices)
        
        # Similarly, create global test dataset and filter for current task.
        global_test_ds = WaferMapDataset(
            file_path=config["dataset"]["path"],
            split="test",
            oversample=False,
            target_dim=(224, 224),
            task_classes=global_classes
        )
        test_indices = np.where(np.isin(np.array(global_test_ds.y), local_classes))[0]
        test_dataset = Subset(global_test_ds, test_indices)

        # Hyperparameter optimization using the global number of classes.
        best_lr, best_wd = tune_hyperparameters_for_task(
            config, train_dataset, model_factory, device,
            num_epochs=config["experiment"]["num_epochs"],
            num_trials=config["experiment"]["num_trials"],
            num_classes=global_num_classes
        )
        print(f"[INFO] Best hyperparameters for Task {task_idx}: lr={best_lr}, weight_decay={best_wd}")

        # Update or initialize the model.
        if task_idx == 0:
            model, criterion, optimizer = model_factory(best_lr, best_wd, global_num_classes, device)
        else:
            old_num_classes = sum(len(task_list[j]) for j in range(task_idx))
            new_classes_count = global_num_classes - old_num_classes
            model = update_model_output(model, new_classes_count=new_classes_count, device=device)
            optimizer = torch.optim.SGD(model.parameters(), lr=best_lr, weight_decay=best_wd)

        # Print model information.
        print_model_info(model, task_idx)

        # Set up TensorBoard logging.
        log_dir = os.path.join(config["logging"]["base_log_dir"], ts, f"task_{task_idx}")
        print(f"[INFO] TensorBoard log directory for Task {task_idx}: {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)

        # Training: if using EWC and task_idx > 0, call train_ewc; otherwise, use standard training.
        if config["experiment"].get("continual_method") == "ewc" and task_idx > 0:
            losses = train_ewc(
                model, train_dataset,
                iters=config["experiment"]["num_epochs"] * len(train_dataset),
                lr=best_lr, batch_size=64,
                device=device, current_task=task_idx,
                ewc_lambda=config["experiment"].get("ewc_lambda", 100.0),
                task_classes=global_classes
            )
        else:
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            for epoch in range(config["experiment"]["num_epochs"]):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
                writer.add_scalar("Train/Loss", train_loss, epoch)
                writer.add_scalar("Validation/Loss", val_loss, epoch)
                print(f"[INFO] Task {task_idx} Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        writer.close()
        print(f"[INFO] Finished Task {task_idx}")

        # Estimate Fisher Information Matrix for EWC.
        if config["experiment"].get("continual_method") == "ewc":
            estimate_fisher(model, train_dataset, n_samples=100, device=device)

        # Save model checkpoint.
        save_model_checkpoint(model, config, task_idx=task_idx, timestamp=ts)

        # Evaluation: Evaluate both current task (filtered) and cumulative test set.
        evaluate_current_and_cumulative(config, model, device, task_list, current_task_index=task_idx)

    print("[INFO] Continual learning training completed.")

def main():
    args = parse_args()
    config = load_config(args.config)
    print("[INFO] Loaded configuration:")
    print(yaml.dump(config, default_flow_style=False))
    if config["experiment"].get("reproducibility", False):
        seed = config["experiment"].get("seed", 42)
        from utils import set_seed
        set_seed(seed)
        print(f"[INFO] Reproducibility enabled. Seed set to {seed}")
    if config["model"]["type"] == "resnet50":
        from Models.Wafer_resnet_model import create_resnet_model
        model_factory = create_resnet_model
    elif config["model"]["type"] == "simplenn":
        from Models.Wafer_simple_model import create_simple_model
        model_factory = create_simple_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    final_model = continual_training_pipeline(config, model_factory, device)
    return

if __name__ == '__main__':
    main()
