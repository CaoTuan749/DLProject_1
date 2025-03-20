#!/usr/bin/env python
"""
Standard Full Dataset Training Pipeline

This script:
  - Loads a YAML configuration file.
  - Performs k-fold cross-validation with hyperparameter tuning (via Optuna).
  - Retrains the final model on the entire training set using the best hyperparameters.
  - Evaluates the final model on a separate test set.
  - Logs metrics (including confusion matrices) to TensorBoard.
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.exceptions import TrialPruned
import matplotlib.pyplot as plt

# Import shared utilities and dataset
from utils import set_seed, save_model_checkpoint
from utils import train_one_epoch, validate_one_epoch, evaluate_model, plot_confusion_matrix, figure_to_tensor, plot_training_curve
from Wafer_data_dataset_resize import WaferMapDataset

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["dataset"]["path"] = os.path.expanduser(config["dataset"]["path"])
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Standard Full Dataset Training Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    return parser.parse_args()

def train_one_fold_standard(model, train_loader, val_loader, criterion, optimizer, writer, fold_idx=0, trial_idx=0, num_epochs=20, early_stopping_patience=2, device="cpu"):
    print(f"[INFO] >>> Starting fold {fold_idx+1} training (Trial #{trial_idx}) for {num_epochs} epochs...")
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Lists to store metrics for plotting curves later
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_one_epoch(model, val_loader, criterion, device)

        train_loss_list.append(t_loss)
        val_loss_list.append(v_loss)
        train_acc_list.append(t_acc)
        val_acc_list.append(v_acc)

        writer.add_scalar(f"Fold{fold_idx}/Train_Loss", t_loss, epoch)
        writer.add_scalar(f"Fold{fold_idx}/Train_Acc", t_acc, epoch)
        writer.add_scalar(f"Fold{fold_idx}/Val_Loss", v_loss, epoch)
        writer.add_scalar(f"Fold{fold_idx}/Val_Acc", v_acc, epoch)
        print(f"[Fold {fold_idx+1} Epoch {epoch+1}/{num_epochs}] Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"[INFO] Early stopping on fold {fold_idx+1} at epoch {epoch+1}")
                break

    # Plot and log training and validation loss curve
    fig_loss = plot_training_curve(train_loss_list, val_loss_list, metric_name="Loss")
    writer.add_image(f"Fold{fold_idx}/Loss_Curve", figure_to_tensor(fig_loss), global_step=fold_idx+1)
    plt.close(fig_loss)

    # Plot and log training and validation accuracy curve
    fig_acc = plot_training_curve(train_acc_list, val_acc_list, metric_name="Accuracy")
    writer.add_image(f"Fold{fold_idx}/Accuracy_Curve", figure_to_tensor(fig_acc), global_step=fold_idx+1)
    plt.close(fig_acc)

    print(f"[INFO] <<< Finished fold {fold_idx+1}, best val loss = {best_val_loss:.4f}\n")
    return best_val_loss

def objective(trial, config, train_dataset_full, model_factory, num_classes, device, num_epochs):
    lr_range = config["experiment"]["suggest"]["lr"]
    wd_range = config["experiment"]["suggest"]["weight_decay"]
    trial_lr = trial.suggest_float('lr',
                                   float(lr_range["low"]),
                                   float(lr_range["high"]),
                                   log=lr_range["log"])
    trial_weight_decay = trial.suggest_float('weight_decay',
                                             float(wd_range["low"]),
                                             float(wd_range["high"]),
                                             log=wd_range["log"])
    print(f"\n[OPTUNA] Starting Trial #{trial.number} with lr={trial_lr:.6f}, weight_decay={trial_weight_decay:.6f}")
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_indices = np.arange(len(train_dataset_full))
    all_labels = train_dataset_full.y
    writer_dir = f"Logs/runs/optuna_trial_{trial.number}_{int(time.time())}"
    writer = SummaryWriter(log_dir=writer_dir)
    fold_accuracies = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
        print(f"[OPTUNA] Trial #{trial.number}: Starting fold {fold_idx+1}/{k_folds}")
        train_subset = Subset(train_dataset_full, train_idx)
        val_subset = Subset(train_dataset_full, val_idx)
        train_loader_fold = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader_fold = DataLoader(val_subset, batch_size=64, shuffle=False)
        model_fold, criterion_fold, optimizer_fold = model_factory(trial_lr, trial_weight_decay, num_classes, device)
        best_val_loss = train_one_fold_standard(model_fold, train_loader_fold, val_loader_fold, criterion_fold, optimizer_fold, writer,
                                                fold_idx=fold_idx, trial_idx=trial.number, num_epochs=num_epochs, early_stopping_patience=2, device=device)
        fold_accuracies.append(1 - best_val_loss)  # converting loss to an accuracy-like measure (lower loss => higher acc)
        
        # Log confusion matrix for each fold
        val_preds, val_labels = evaluate_model(model_fold, val_loader_fold, device)
        cm = confusion_matrix(val_labels, val_preds)
        fig_cm = plot_confusion_matrix(cm, train_dataset_full.encoder.classes_)
        writer.add_image(f"Fold_{fold_idx+1}_Confusion_Matrix", figure_to_tensor(fig_cm), global_step=fold_idx+1)
        plt.close(fig_cm)
        
    writer.close()
    avg_acc = np.mean(fold_accuracies)
    print(f"[OPTUNA] Trial #{trial.number} done. Fold Accuracies: {fold_accuracies}. Avg Acc={avg_acc:.4f}")
    trial.report(avg_acc, step=k_folds)
    if trial.should_prune():
        raise TrialPruned()
    return avg_acc

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

    num_trials = config["experiment"]["num_trials"]
    num_epochs = config["experiment"]["num_epochs"]
    final_epochs = config["experiment"]["final_epochs"]
    dataset_path = config["dataset"]["path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_dataset_full = WaferMapDataset(file_path=dataset_path, split="train", oversample=False, target_dim=(224, 224))
    test_dataset = WaferMapDataset(file_path=dataset_path, split="test", oversample=False, target_dim=(224, 224))
    encoder_classes = train_dataset_full.encoder.classes_
    num_classes = len(encoder_classes)
    print(f"[INFO] Number of classes: {num_classes}")
    print("[INFO] Classes:", encoder_classes)

    study_name = f"resnet_wafer_{int(time.time())}"
    db_url = "sqlite:///resnet_wafer_v2.db"
    print(f"[INFO] Creating Optuna study '{study_name}' with DB file: {db_url}")
    study = optuna.create_study(study_name=study_name, storage=db_url, load_if_exists=False, direction='maximize')
    print(f"[INFO] Starting study.optimize with {num_trials} trials...\n")
    study.optimize(lambda trial: objective(trial, config, train_dataset_full, model_factory, num_classes, device, num_epochs), n_trials=num_trials)
    best_trial = study.best_trial
    print("[OPTUNA] Best trial found:")
    print(f"  Trial number: {best_trial.number}")
    print(f"  Avg k-fold acc: {best_trial.value:.4f}")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")
    best_lr = best_trial.params['lr']
    best_weight_decay = best_trial.params['weight_decay']

    print("\n[INFO] Retraining final model on entire training set using best hyperparameters...")
    final_model, final_criterion, final_optimizer = model_factory(best_lr, best_weight_decay, num_classes, device)
    full_train_loader = DataLoader(train_dataset_full, batch_size=64, shuffle=True)
    for epoch in range(final_epochs):
        t_loss, t_acc = train_one_epoch(final_model, full_train_loader, final_criterion, final_optimizer, device)
        print(f"[Final Train] Epoch {epoch+1}/{final_epochs} - Loss: {t_loss:.4f}, Acc: {t_acc:.4f}")

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("\n[INFO] Evaluating final model on test set...")
    all_preds, all_labels = evaluate_model(final_model, test_loader, device)
    test_accuracy = (all_preds == all_labels).mean()
    print(f"[RESULT] Final Test Accuracy: {test_accuracy:.4f}\n")
    from sklearn.metrics import classification_report
    print("[RESULT] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=encoder_classes))
    
    if config["experiment"].get("save_model", False):
        save_model_checkpoint(final_model, config)
    
    print("\n[INFO] Done. Logs are successfully stored in TensorBoard and the Optuna dashboard.")

if __name__ == '__main__':
    main()