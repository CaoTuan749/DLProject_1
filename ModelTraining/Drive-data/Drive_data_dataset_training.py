#!/usr/bin/env python

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import optuna
from optuna.exceptions import TrialPruned

# Import the drive dataset
from Drive_data_dataset import SMARTDataset

# Import utilities from utils.py
from utils import load_config, get_tensorboard_writer

# ------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a drive regression model using YAML configuration with k-fold CV and Optuna."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)."
    )
    return parser.parse_args()

# ------------------------------------------------------------
# Helper Functions for Training, Validation, and Evaluation (Regression)
# ------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    mse = mean_squared_error(all_targets, all_preds)
    return np.array(all_preds), np.array(all_targets), mse

# ------------------------------------------------------------
# Data Loading and Splitting
# ------------------------------------------------------------
def load_and_split_datasets(config):
    """
    Loads the full drive dataset and returns the training and test sets.
    The dataset path is read from config["dataset"]["path"].
    """
    dataset_path = config["dataset"]["path"]
    
    # Create the full training dataset.
    train_dataset_full = SMARTDataset(data_directory=dataset_path)
    
    # For this example, assume that the test dataset is created separately.
    # (If SMARTDataset supports a split flag, adjust accordingly.)
    test_dataset = SMARTDataset(data_directory=dataset_path)
    
    print(f"[INFO] Training dataset samples: {len(train_dataset_full)}")
    print(f"[INFO] Test dataset samples: {len(test_dataset)}")
    
    return train_dataset_full, test_dataset

# ------------------------------------------------------------
# Main Training Pipeline
# ------------------------------------------------------------
def main():
    # Parse command-line arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    print("Loaded configuration:")
    print(config)
    
    # ------------------------------------------------------------------
    # Set up model factory based on YAML config.
    # For example, if config["model"]["type"] is "simplenn", import from Models/Drive_simple_model.py.
    # ------------------------------------------------------------------
    if config["model"]["type"] == "simplenn":
        from Models.simple_nn import create_simple_model
        model_factory = create_simple_model
    else:
        raise ValueError("Unknown model type specified in config.")
    
    # Retrieve experiment settings from configuration.
    num_trials = config["experiment"]["num_trials"]
    num_epochs = config["experiment"]["num_epochs"]
    final_epochs = config["experiment"]["final_epochs"]
    batch_size = config["experiment"]["batch_size"]
    k_folds = config["experiment"]["k_folds"]
    early_stopping_patience = config["experiment"]["early_stopping_patience"]
    # Fixed hidden dimension from config (not tuned)
    fixed_hidden_dim = config["model"]["hidden_dim"]
    
    # Set up device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load datasets
    train_dataset_full, test_dataset = load_and_split_datasets(config)
    
    # ------------------------------------------------------------
    # Define a Single-Fold Training Routine
    # ------------------------------------------------------------
    def train_one_fold(model, train_loader, val_loader, criterion, optimizer, writer,
                       fold_idx=0, trial_idx=0, num_epochs=num_epochs):
        print(f"[INFO] >>> Starting fold {fold_idx+1} training (Trial #{trial_idx}) for {num_epochs} epochs...")
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate_one_epoch(model, val_loader, criterion, device)
            writer.add_scalar(f"Fold{fold_idx}/Train_Loss", train_loss, epoch)
            writer.add_scalar(f"Fold{fold_idx}/Val_Loss", val_loss, epoch)
            print(f"[Fold {fold_idx+1} Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"[INFO] Early stopping on fold {fold_idx+1} at epoch {epoch+1}")
                    break
        print(f"[INFO] <<< Finished fold {fold_idx+1}, best val loss = {best_val_loss:.4f}\n")
        return best_val_loss
    
    # ------------------------------------------------------------
    # Define the Optuna Objective Function (for k-fold cross-validation)
    # ------------------------------------------------------------
    def objective(trial):
        # Suggest hyperparameters: learning rate and weight decay.
        trial_lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        trial_weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        print(f"\n[OPTUNA] Starting Trial #{trial.number} with lr={trial_lr:.6f}, weight_decay={trial_weight_decay:.6f}")
        
        # Set up k-fold cross validation.
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        all_indices = np.arange(len(train_dataset_full))
        writer_dir = os.path.join(config["logging"]["tensorboard_log_dir"],
                                  f"optuna_trial_{trial.number}_{int(time.time())}")
        writer = get_tensorboard_writer(writer_dir)
        fold_losses = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
            print(f"[OPTUNA] Trial #{trial.number}: Starting fold {fold_idx+1}/{k_folds}")
            train_subset = Subset(train_dataset_full, train_idx)
            val_subset = Subset(train_dataset_full, val_idx)
            train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader_fold = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            input_dim = train_dataset_full.X.shape[1]
            # Create a new model instance for this fold using trial hyperparameters.
            model_fold = model_factory(trial_lr, trial_weight_decay, input_dim, fixed_hidden_dim, output_dim=1).to(device)
            criterion_fold = torch.nn.MSELoss()
            optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=trial_lr, weight_decay=trial_weight_decay)
            best_val_loss = train_one_fold(
                model_fold,
                train_loader_fold,
                val_loader_fold,
                criterion_fold,
                optimizer_fold,
                writer,
                fold_idx=fold_idx,
                trial_idx=trial.number,
                num_epochs=num_epochs
            )
            fold_losses.append(best_val_loss)
        writer.close()
        avg_loss = np.mean(fold_losses)
        print(f"[OPTUNA] Trial #{trial.number} done. Fold Losses: {fold_losses}. Avg Loss={avg_loss:.4f}")
        trial.report(avg_loss, step=k_folds)
        if trial.should_prune():
            raise TrialPruned()
        return avg_loss
    
    # ------------------------------------------------------------
    # Create an Optuna study and run the optimization
    # ------------------------------------------------------------
    study_name = f"drive_regression_{int(time.time())}"
    db_url = f"sqlite:///{study_name}.db"
    print(f"[INFO] Creating Optuna study '{study_name}' with DB file: {db_url}")
    study = optuna.create_study(study_name=study_name, storage=db_url, load_if_exists=False, direction='minimize')
    print(f"[INFO] Starting study.optimize with {num_trials} trials...\n")
    study.optimize(objective, n_trials=num_trials)
    best_trial = study.best_trial
    print("[OPTUNA] Best trial found:")
    print(f"  Trial number: {best_trial.number}")
    print(f"  Avg k-fold loss: {best_trial.value:.4f}")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")
    best_lr = best_trial.params['lr']
    best_weight_decay = best_trial.params['weight_decay']
    
    # ------------------------------------------------------------
    # Retrain the Final Model on the Full Training Set using Best Hyperparameters
    # ------------------------------------------------------------
    print("\n[INFO] Retraining final model on the entire training set using best hyperparameters...")
    input_dim = train_dataset_full.X.shape[1]
    final_model = model_factory(best_lr, best_weight_decay, input_dim, fixed_hidden_dim, output_dim=1).to(device)
    final_criterion = torch.nn.MSELoss()
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    final_train_loader = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True)
    writer_final = get_tensorboard_writer(os.path.join(config["logging"]["tensorboard_log_dir"],
                                                         f"final_training_{int(time.time())}"))
    for epoch in range(final_epochs):
        train_loss = train_one_epoch(final_model, final_train_loader, final_criterion, final_optimizer, device)
        writer_final.add_scalar("Final_Train_Loss", train_loss, epoch)
        print(f"[Final Train] Epoch {epoch+1}/{final_epochs} - Loss: {train_loss:.4f}")
    writer_final.close()
    
    # ------------------------------------------------------------
    # Evaluate the Final Model on the Test Set
    # ------------------------------------------------------------
    final_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_targets, test_mse = evaluate_model(final_model, final_test_loader, device)
    print(f"\n[RESULT] Final Test MSE: {test_mse:.4f}\n")
    print("[INFO] Done. You can now view logs in TensorBoard and the Optuna dashboard.")
    
if __name__ == '__main__':
    main()
