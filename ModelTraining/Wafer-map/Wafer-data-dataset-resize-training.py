#!/usr/bin/env python
"""
train_wafer_optuna_kfold_tensorboard.py

This script performs training on wafer map data with the following steps:
  - Loads a YAML configuration file that contains hyperparameters and experiment settings.
  - Sets up the device (CPU/GPU) and loads the dataset.
  - Performs k-fold cross-validation with Optuna hyperparameter optimization.
  - Retrains the final model on the entire training set using the best hyperparameters.
  - Evaluates the final model on a separate test set.
  - Logs metrics to TensorBoard.
"""

import os
import time
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.exceptions import TrialPruned

# Import dataset
from Wafer_data_dataset_resize import WaferMapDataset

# ------------------------------------------------------------
# Utility: Load YAML Configuration
# ------------------------------------------------------------
def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file and returns a dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Expand tilde in dataset path if present.
    config["dataset"]["path"] = os.path.expanduser(config["dataset"]["path"])
    return config

# ------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a wafer map classification model using YAML configuration."
    )
    # Add a config file argument
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)."
    )
    return parser.parse_args()

# ------------------------------------------------------------
# Helper Functions for Training, Validation, and Evaluation
# ------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Expand single-channel images to 3 channels (if needed by the model)
        inputs = inputs.repeat(1, 3, 1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss / len(dataloader), correct / total

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.repeat(1, 3, 1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(dataloader), correct / total

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.repeat(1, 3, 1, 1)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

# ------------------------------------------------------------
# Main Training Pipeline
# ------------------------------------------------------------
def main():
    # Parse command-line arguments and load the YAML configuration
    args = parse_args()
    config = load_config(args.config)
    print("Loaded configuration:")
    print(config)

    # Set up model factory based on the YAML config
    if config["model"]["type"] == "resnet50":
        from Models.Wafer_resnet_model import create_resnet_model
        model_factory = create_resnet_model
    elif config["model"]["type"] == "simplenn":
        from Models.Wafer_simple_model import create_simple_model
        model_factory = create_simple_model

    # Use configuration settings for hyperparameters and experiment parameters
    num_trials = config["experiment"]["num_trials"]
    num_epochs = config["experiment"]["num_epochs"]
    final_epochs = config["experiment"]["final_epochs"]
    lr = config["model"]["lr"]
    weight_decay = config["model"]["weight_decay"]
    dataset_path = config["dataset"]["path"]

    # Set up device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load datasets
    train_dataset_full = WaferMapDataset(
        file_path=dataset_path,
        split="train",
        oversample=False,
        target_dim=(224, 224)
    )
    test_dataset = WaferMapDataset(
        file_path=dataset_path,
        split="test",
        oversample=False,
        target_dim=(224, 224)
    )
    encoder_classes = train_dataset_full.encoder.classes_
    num_classes = len(encoder_classes)
    print(f"[INFO] Number of classes: {num_classes}")
    print("[INFO] Classes:", encoder_classes)

    # ------------------------------------------------------------
    # Define a Single-Fold Training Routine using helper functions
    # ------------------------------------------------------------
    def train_one_fold(model, train_loader, val_loader, criterion, optimizer, writer,
                       fold_idx=0, trial_idx=0, num_epochs=num_epochs, early_stopping_patience=2):
        print(f"[INFO] >>> Starting fold {fold_idx+1} training (Trial #{trial_idx}) for {num_epochs} epochs...")
        best_val_loss = float('inf')
        best_val_acc = 0.0
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            writer.add_scalar(f"Fold{fold_idx}/Train_Loss", train_loss, epoch)
            writer.add_scalar(f"Fold{fold_idx}/Train_Acc", train_acc, epoch)
            writer.add_scalar(f"Fold{fold_idx}/Val_Loss", val_loss, epoch)
            writer.add_scalar(f"Fold{fold_idx}/Val_Acc", val_acc, epoch)
            print(f"[Fold {fold_idx+1} Epoch {epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"[INFO] Early stopping on fold {fold_idx+1} at epoch {epoch+1}")
                    break
            best_val_acc = max(best_val_acc, val_acc)
        print(f"[INFO] <<< Finished fold {fold_idx+1}, best val acc = {best_val_acc:.4f}\n")
        return best_val_acc

    # ------------------------------------------------------------
    # Define the Optuna Objective Function (for k-fold cross-validation)
    # ------------------------------------------------------------
    def objective(trial):
        # Suggest hyperparameters (you can override YAML values if desired)
        trial_lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        trial_weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        print(f"\n[OPTUNA] Starting Trial #{trial.number} with lr={trial_lr:.6f}, weight_decay={trial_weight_decay:.6f}")
        k_folds = 5
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        all_indices = np.arange(len(train_dataset_full))
        all_labels = train_dataset_full.y
        writer_dir = f"runs/optuna_trial_{trial.number}_{int(time.time())}"
        writer = SummaryWriter(log_dir=writer_dir)
        fold_accuracies = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
            print(f"[OPTUNA] Trial #{trial.number}: Starting fold {fold_idx+1}/{k_folds}")
            train_subset = Subset(train_dataset_full, train_idx)
            val_subset = Subset(train_dataset_full, val_idx)
            batch_size_fold = 64
            train_loader_fold = DataLoader(train_subset, batch_size=batch_size_fold, shuffle=True)
            val_loader_fold = DataLoader(val_subset, batch_size=batch_size_fold, shuffle=False)
            model_fold, criterion_fold, optimizer_fold = model_factory(trial_lr, trial_weight_decay, num_classes, device)
            best_val_acc = train_one_fold(
                model_fold,
                train_loader_fold,
                val_loader_fold,
                criterion_fold,
                optimizer_fold,
                writer,
                fold_idx=fold_idx,
                trial_idx=trial.number,
                num_epochs=num_epochs,
                early_stopping_patience=2
            )
            fold_accuracies.append(best_val_acc)
        writer.close()
        avg_acc = np.mean(fold_accuracies)
        print(f"[OPTUNA] Trial #{trial.number} done. Fold Accuracies: {fold_accuracies}. Avg Acc={avg_acc:.4f}")
        trial.report(avg_acc, step=k_folds)
        if trial.should_prune():
            raise TrialPruned()
        return avg_acc

    # ------------------------------------------------------------
    # Create an Optuna study and run the optimization
    # ------------------------------------------------------------
    study_name = f"resnet_wafer_{int(time.time())}"
    db_url = "sqlite:///resnet_wafer_v2.db"
    print(f"[INFO] Creating Optuna study '{study_name}' with DB file: resnet_wafer.db")
    study = optuna.create_study(study_name=study_name, storage=db_url, load_if_exists=False, direction='maximize')
    print(f"[INFO] Starting study.optimize with {num_trials} trials...\n")
    study.optimize(objective, n_trials=num_trials)
    best_trial = study.best_trial
    print("[OPTUNA] Best trial found:")
    print(f"  Trial number: {best_trial.number}")
    print(f"  Avg k-fold acc: {best_trial.value:.4f}")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")
    best_lr = best_trial.params['lr']
    best_weight_decay = best_trial.params['weight_decay']

    # ------------------------------------------------------------
    # Retrain the Final Model on the Full Training Set using Best Hyperparameters
    # ------------------------------------------------------------
    print("\n[INFO] Retraining final model on entire training set using best hyperparams...")
    final_model, final_criterion, final_optimizer = model_factory(best_lr, best_weight_decay, num_classes, device)
    full_train_loader = DataLoader(train_dataset_full, batch_size=64, shuffle=True)
    for epoch in range(final_epochs):
        train_loss, train_acc = train_one_epoch(final_model, full_train_loader, final_criterion, final_optimizer, device)
        print(f"[Final Train] Epoch {epoch+1}/{final_epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    # ------------------------------------------------------------
    # Evaluate the Final Model on the Test Set
    # ------------------------------------------------------------
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("\n[INFO] Evaluating final model on test set...")
    all_preds, all_labels = evaluate_model(final_model, test_loader, device)
    test_accuracy = (all_preds == all_labels).mean()
    print(f"[RESULT] Final Test Accuracy: {test_accuracy:.4f}\n")
    print("[RESULT] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=encoder_classes))
    print("\n[INFO] Done. You can now view logs in TensorBoard and the Optuna dashboard.")

if __name__ == '__main__':
    main()
