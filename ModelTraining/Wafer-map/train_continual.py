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
      * Updates the model’s output layer to accommodate new classes.
      * Evaluates the model on:
            - The current task’s test dataset.
            - The cumulative test dataset (all tasks up to the current one).
      * Logs training/validation curves, GPU memory usage, and confusion matrices to TensorBoard.
  - Returns the final trained model.
"""

import os
import time
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.exceptions import TrialPruned
import matplotlib.pyplot as plt


# Import shared utilities and dataset class.
from utils import set_seed, save_model_checkpoint, update_model_output, plot_confusion_matrix, figure_to_tensor
from utils import train_one_epoch, validate_one_epoch, evaluate_model
from Wafer_data_dataset_resize import WaferMapDataset

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["dataset"]["path"] = os.path.expanduser(config["dataset"]["path"])
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning Training Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    return parser.parse_args()

def tune_hyperparameters_for_task(config, task_dataset, model_factory, device, num_classes, num_epochs, num_trials):
    """
    Performs HPO (via Optuna) on the provided task dataset and returns the best lr and weight decay.
    Evaluates based on the average validation loss.
    """
    from sklearn.model_selection import StratifiedKFold
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
        all_labels = task_dataset.y
        fold_losses = []
        for train_idx, val_idx in skf.split(all_indices, all_labels):
            train_subset = torch.utils.data.Subset(task_dataset, train_idx)
            val_subset = torch.utils.data.Subset(task_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
            model, criterion, optimizer = model_factory(trial_lr, trial_weight_decay, num_classes, device)
            best_val_loss = float('inf')
            for epoch in range(num_epochs):
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
      (a) The current task's test set.
      (b) The cumulative test set from tasks 0 to current_task_index.
    Logs the results to TensorBoard and prints them.
    """
    from Wafer_data_dataset_resize import WaferMapDataset
    from sklearn.metrics import classification_report
    # Evaluate current task test set.
    current_test_ds = WaferMapDataset(
        file_path=config["dataset"]["path"],
        split="test",
        oversample=False,
        target_dim=(224, 224),
        task_classes=task_list[current_task_index]
    )
    current_loader = DataLoader(current_test_ds, batch_size=64, shuffle=False)
    preds, labels = evaluate_model(model, current_loader, device)
    current_acc = (preds == labels).mean()
    current_report = classification_report(labels, preds, digits=3)
    print(f"[Evaluation] Task {current_task_index} test set accuracy: {current_acc:.4f}")
    print(f"[Evaluation] Classification Report for Task {current_task_index}:\n{current_report}")
    
    # For cumulative evaluation, combine test data from tasks 0 to current_task_index.
    cumulative_X, cumulative_y = [], []
    for t in range(current_task_index + 1):
        ds = WaferMapDataset(
            file_path=config["dataset"]["path"],
            split="test",
            oversample=False,
            target_dim=(224, 224),
            task_classes=task_list[t]
        )
        cumulative_X.append(ds.X)
        cumulative_y.append(ds.y)
    cum_X = np.concatenate(cumulative_X, axis=0)
    cum_y = np.concatenate(cumulative_y, axis=0)
    cum_tensor_X = torch.tensor(cum_X).view(-1, 1, 224, 224).repeat(1, 3, 1, 1)
    cum_tensor_y = torch.tensor(cum_y)
    cum_dataset = torch.utils.data.TensorDataset(cum_tensor_X, cum_tensor_y)
    cum_loader = DataLoader(cum_dataset, batch_size=64, shuffle=False)
    cum_preds, cum_labels = evaluate_model(model, cum_loader, device)
    cum_acc = (cum_preds == cum_labels).mean()
    cum_report = classification_report(cum_labels, cum_preds, digits=3)
    print(f"[Evaluation] Cumulative test set accuracy (tasks 0 to {current_task_index}): {cum_acc:.4f}")
    print(f"[Evaluation] Cumulative Classification Report (tasks 0 to {current_task_index}):\n{cum_report}")
    
    # Log results to TensorBoard
    eval_log_dir = os.path.join("Logs", "evaluation", f"task{current_task_index}")
    writer = SummaryWriter(log_dir=eval_log_dir)
    writer.add_scalar("Test/Current_Accuracy", current_acc, global_step=current_task_index)
    writer.add_text("Test/Current_Classification_Report", current_report, global_step=current_task_index)
    writer.add_scalar("Test/Cumulative_Accuracy", cum_acc, global_step=current_task_index)
    writer.add_text("Test/Cumulative_Classification_Report", cum_report, global_step=current_task_index)
    writer.close()

def continual_training_pipeline(config, model_factory, device, original_num_classes):
    print("[INFO] Running continual learning training with HPO per task...")
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_log_dir = config["experiment"].get("tensorboard_log_dir", "runs")
    method = config["experiment"].get("continual_method", "baseline")
    task_list = config["experiment"].get("task_list", [])
    if not task_list:
        raise ValueError("No task_list provided in config for continual learning.")

    # --- Task 0 ---
    print(f"[INFO] Starting Task 0 with classes: {task_list[0]}")
    train_dataset = WaferMapDataset(file_path=config["dataset"]["path"],
                                    split="train",
                                    oversample=False,
                                    target_dim=(224, 224),
                                    task_classes=task_list[0])
    test_dataset = WaferMapDataset(file_path=config["dataset"]["path"],
                                   split="test",
                                   oversample=False,
                                   target_dim=(224, 224),
                                   task_classes=task_list[0])
    num_task_classes = len(task_list[0])
    best_lr, best_wd = tune_hyperparameters_for_task(config, train_dataset, model_factory, device, num_task_classes, config["experiment"]["num_epochs"], config["experiment"]["num_trials"])
    print(f"[INFO] Best hyperparameters for Task 0: lr={best_lr}, weight_decay={best_wd}")
    model, criterion, optimizer = model_factory(best_lr, best_wd, num_task_classes, device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    log_dir = os.path.join(base_log_dir, method, ts, "task0")
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(config["experiment"]["num_epochs"]):
         t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
         v_loss, v_acc = validate_one_epoch(model, test_loader, criterion, device)
         writer.add_scalar("Train/Loss", t_loss, epoch)
         writer.add_scalar("Train/Accuracy", t_acc, epoch)
         writer.add_scalar("Validation/Loss", v_loss, epoch)
         writer.add_scalar("Validation/Accuracy", v_acc, epoch)
         if torch.cuda.is_available():
             mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
             writer.add_scalar("GPU/MemoryAllocated_MB", mem_alloc, epoch)
         print(f"[Task 0 Final Epoch {epoch+1}] Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
    writer.close()
    save_model_checkpoint(model, config, task_idx=0, timestamp=ts)
    # (Optional) Estimate Fisher information after Task 0 if using EWC.
    if method.lower() == "ewc":
        from ewc import estimate_fisher
        # For example, use 100 samples for Fisher estimation.
        estimate_fisher(model, train_dataset, n_samples=100)

    # --- For Task 1 and beyond ---
    for t in range(1, len(task_list)):
        print(f"[INFO] Starting Task {t} with new classes: {task_list[t]}")
        new_classes_count = len(task_list[t])
        model = update_model_output(model, new_classes_count=new_classes_count, device=device)
        current_num_classes = model.fc.out_features
        train_dataset = WaferMapDataset(file_path=config["dataset"]["path"],
                                        split="train",
                                        oversample=False,
                                        target_dim=(224, 224),
                                        task_classes=task_list[t])
        test_dataset = WaferMapDataset(file_path=config["dataset"]["path"],
                                       split="test",
                                       oversample=False,
                                       target_dim=(224, 224),
                                       task_classes=task_list[t])
        best_lr, best_wd = tune_hyperparameters_for_task(config, train_dataset, model_factory, device, current_num_classes, config["experiment"]["num_epochs"], config["experiment"]["num_trials"])
        print(f"[INFO] Best hyperparameters for Task {t}: lr={best_lr}, weight_decay={best_wd}")
        # Update optimizer with tuned hyperparameters (keep model weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=best_lr, weight_decay=best_wd)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        log_dir = os.path.join(base_log_dir, method, ts, f"task{t}")
        writer = SummaryWriter(log_dir=log_dir)
        if method.lower() == "ewc":
            # Use the EWC training function
            from ewc import train_ewc
            # Define number of iterations (e.g., num_epochs * number of batches)
            iters = config["experiment"]["num_epochs"] * len(train_loader)
            # Use ewc_lambda from config if provided; default to 100.0
            ewc_lambda = config["experiment"].get("ewc_lambda", 100.0)
            # Here, we assume task_id equals t (adjust if necessary)
            losses = train_ewc(model, train_dataset, iters, best_lr, batch_size=64, current_task=t+1, ewc_lambda=ewc_lambda, task_id=t, verbose=True)
        else:
            for epoch in range(config["experiment"]["num_epochs"]):
                t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                v_loss, v_acc = validate_one_epoch(model, test_loader, criterion, device)
                writer.add_scalar("Train/Loss", t_loss, epoch)
                writer.add_scalar("Train/Accuracy", t_acc, epoch)
                writer.add_scalar("Validation/Loss", v_loss, epoch)
                writer.add_scalar("Validation/Accuracy", v_acc, epoch)
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    writer.add_scalar("GPU/MemoryAllocated_MB", mem_alloc, epoch)
                print(f"[Task {t} Final Epoch {epoch+1}] Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        writer.close()
        save_model_checkpoint(model, config, task_idx=t, timestamp=ts)
        # If using EWC, update Fisher information after training on the current task
        if method.lower() == "ewc":
            from ewc import estimate_fisher
            estimate_fisher(model, train_dataset, n_samples=100)
        # Evaluate after finishing each task
        from utils import evaluate_current_and_cumulative
        evaluate_current_and_cumulative(config, model, device, task_list, current_task_index=t)
    print("[INFO] Continual learning training complete.")
    return model


def main():
    args = parse_args()
    config = load_config(args.config)
    print("Loaded configuration:")
    print(config)
    if config["experiment"].get("reproducibility", False):
        seed = config["experiment"].get("seed", 42)
        set_seed(seed)
        print(f"[INFO] Reproducibility enabled. Seed set to {seed}")
    if config["model"]["type"] == "resnet50":
        from Models.Wafer_resnet_model import create_resnet_model
        model_factory = create_resnet_model
    elif config["model"]["type"] == "simplenn":
        from Models.Wafer_simple_model import create_simple_model
        model_factory = create_simple_model

    dataset_path = config["dataset"]["path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    train_dataset_full = WaferMapDataset(file_path=dataset_path, split="train", oversample=False, target_dim=(224, 224))
    encoder_classes = train_dataset_full.encoder.classes_
    num_classes = len(encoder_classes)
    print(f"[INFO] Number of classes: {num_classes}")
    print("[INFO] Classes:", encoder_classes)
    
    final_model = continual_training_pipeline(config, model_factory, device, num_classes)
    return

if __name__ == '__main__':
    main()
