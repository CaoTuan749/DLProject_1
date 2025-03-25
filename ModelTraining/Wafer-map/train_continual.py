#!/usr/bin/env python
"""
Continual Learning Training Pipeline

This script performs the following:
  - Loads a YAML configuration file.
  - Runs a continual learning pipeline with per-task hyperparameter optimization (HPO).
  - For each task:
      * Tunes hyperparameters on task-specific training data.
      * Retrains the model on the entire task using the best hyperparameters.
      * Saves a checkpoint.
      * [Currently disabled] Updates the model’s output layer to accommodate new classes.
      * Evaluates the model on:
            - The current task’s test dataset (filtered to only current task classes).
            - The cumulative test dataset (all tasks up to the current one, using global mapping).
      * Logs training/validation curves, confusion matrices, and classification reports to TensorBoard.
  - Returns the final trained model.
"""

import os
import time
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.exceptions import TrialPruned

# Import shared utilities and dataset class.
from utils import (
    train_one_epoch, validate_one_epoch, evaluate_model, save_model_checkpoint,
    update_model_output, apply_mask, print_model_info, parse_args, load_config,
    figure_to_tensor, plot_confusion_matrix
)
from Wafer_data_dataset_resize import WaferMapDataset
# Import the EWC functions (assumed to be defined in a separate file 'ewc.py')
from ewc import estimate_fisher, train_ewc

# ------------------------------------------------------------------------------
# SubDataset Class
# This class creates a subset of the original dataset based on provided labels.
class SubDataset(Dataset):
    def __init__(self, original_dataset, sub_labels, target_transform=None):
        """
        Args:
            original_dataset: The full dataset (assumed to have a 'y' attribute for labels).
            sub_labels (list): List of labels to keep.
            target_transform (callable, optional): A function to transform the label.
        """
        super().__init__()
        self.dataset = original_dataset
        # Convert sub_labels to a tensor (assuming integer labels)
        self.sub_labels = torch.tensor(sub_labels, dtype=torch.long)
        # Create a boolean mask for filtering dataset samples with labels in sub_labels
        mask = torch.isin(torch.tensor(self.dataset.y), self.sub_labels)
        self.sub_indices = torch.nonzero(mask).squeeze().tolist()
        # Store the filtered labels as an attribute 'y'
        if isinstance(self.sub_indices, int):
            self.y = [self.dataset.y[self.sub_indices]]
        else:
            self.y = np.array(self.dataset.y)[self.sub_indices].tolist()
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        # Fetch the sample from the original dataset using the filtered index
        sample = self.dataset[self.sub_indices[index]]
        # If a target_transform is provided, transform the label accordingly
        if self.target_transform:
            original_label = sample[1]
            target = self.target_transform(original_label)
            # Uncomment the following line if you wish to debug label transformations.
            # print(f"Original label: {original_label}, Transformed label: {target}")
            sample = (sample[0], target)
        return sample

# ------------------------------------------------------------------------------
# Hyperparameter Tuning Function for Each Task
def tune_hyperparameters_for_task(config, task_dataset, model_factory, device,
                                  num_epochs, num_trials, num_classes, task_list,
                                  task_idx, local_classes):
    """
    Performs hyperparameter optimization (HPO) for a given task using cross-validation.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix, classification_report

    # Helper function to extract labels from a dataset
    def get_labels(ds):
        if hasattr(ds, 'y'):
            return np.array(ds.y)
        elif isinstance(ds, Subset):
            return np.array(ds.dataset.y)[ds.indices]
        else:
            raise AttributeError("Dataset does not have attribute 'y'")

    # Compute global classes and create mapping
    global_classes = [cls for sublist in task_list[:task_idx+1] for cls in sublist]
    global_class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}
    target_transform = lambda y: global_class_to_idx[int(y)]
    
    # Map training dataset to only include samples for current local classes
    train_dataset_mapped = SubDataset(
        original_dataset=task_dataset.dataset,
        sub_labels=local_classes,
        target_transform=target_transform
    )
    all_labels = get_labels(train_dataset_mapped)
    all_indices = np.arange(len(train_dataset_mapped))
    skf = StratifiedKFold(n_splits=min(3, len(set(all_labels))), shuffle=True, random_state=42)

    # Create an Optuna study to optimize hyperparameters
    study = optuna.create_study(direction='minimize')

    # Define the objective function for HPO
    def objective(trial):
        trial_lr = trial.suggest_float(
            'lr',
            float(config["experiment"]["suggest"]["lr"]["low"]),
            float(config["experiment"]["suggest"]["lr"]["high"]),
            log=config["experiment"]["suggest"]["lr"]["log"]
        )
        trial_weight_decay = trial.suggest_float(
            'weight_decay',
            float(config["experiment"]["suggest"]["weight_decay"]["low"]),
            float(config["experiment"]["suggest"]["weight_decay"]["high"]),
            log=config["experiment"]["suggest"]["weight_decay"]["log"]
        )
        fold_losses = []
        # Cross-validation loop over folds.
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
            train_subset = Subset(train_dataset_mapped, train_idx)
            val_subset = Subset(train_dataset_mapped, val_idx)
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
            model, criterion, optimizer = model_factory(
                trial_lr, trial_weight_decay, num_classes=len(global_classes), device=device
            )
            best_val_loss = float('inf')
            # Create a timestamp in the format "HHMMddMM" (e.g., "15300506")
            ts = config["experiment"].get("timestamp", time.strftime("%H%M%d%m"))
            base_log_dir = config["logging"]["base_log_dir"]
            # Build a readable log directory name for HPO logs.
            hpo_log_dir = os.path.join(base_log_dir, f"{ts}_Task{task_idx}_HPO_trial{trial.number}_fold{fold_idx}")
            writer = SummaryWriter(log_dir=hpo_log_dir)
            for epoch in range(num_epochs):
                epoch_start = time.time()
                train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, _ = validate_one_epoch(model, val_loader, criterion, device)
                epoch_duration = time.time() - epoch_start
                writer.add_scalar("Train/Loss", train_loss, epoch)
                writer.add_scalar("Validation/Loss", val_loss, epoch)
                print(f"[HPO] Trial {trial.number} Fold {fold_idx} Epoch {epoch}: "
                      f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Time={epoch_duration:.1f}s")
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    writer.close()
                    raise optuna.TrialPruned()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            preds, labels = evaluate_model(model, val_loader, device)
            cm = confusion_matrix(labels, preds)
            fig_cm = plot_confusion_matrix(cm, class_names=[str(cls) for cls in np.unique(labels)])
            cm_tensor = figure_to_tensor(fig_cm)
            writer.add_image("Confusion Matrix", cm_tensor, 0)
            report = classification_report(labels, preds, digits=3)
            writer.add_text("Classification Report", report, 0)
            writer.close()
            fold_losses.append(best_val_loss)
        avg_loss = np.mean(fold_losses)
        print(f"[HPO] Trial {trial.number}: Average Loss={avg_loss:.4f}")
        return avg_loss

    study.optimize(objective, n_trials=num_trials)
    best_trial = study.best_trial
    print(f"[Debug] Best trial: lr={best_trial.params['lr']}, weight_decay={best_trial.params['weight_decay']}")
    return best_trial.params['lr'], best_trial.params['weight_decay']

# ------------------------------------------------------------------------------
# Test Functions
def test_acc(model, dataset, device, task_id=None, test_size=None, batch_size=512):
    """
    Computes the accuracy on a given dataset.
    If task_id is provided, masks outputs for the current task.
    """
    mode = model.training
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    total_tested = 0
    total_correct = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        if test_size is not None and total_tested >= test_size:
            break
        with torch.no_grad():
            scores = model(x)
        if task_id is not None:
            masked_outputs = apply_mask(scores, task_id)
            adjusted_labels = y - 2 * task_id
            _, predicted = torch.max(masked_outputs[:, 2*task_id:2*(task_id+1)], 1)
            total_correct += (predicted == adjusted_labels).sum().item()
        else:
            if hasattr(model, 'fc'):
                num_classes = model.fc.out_features
            elif hasattr(model, 'classifier'):
                last_layer = list(model.classifier.children())[-1]
                num_classes = last_layer.out_features if hasattr(last_layer, 'out_features') else scores.shape[1]
            else:
                num_classes = scores.shape[1]
            adjusted_labels = y % num_classes
            _, predicted = torch.max(scores, 1)
            total_correct += (predicted == adjusted_labels).sum().item()
        total_tested += x.size(0)
    accuracy = total_correct * 100 / total_tested
    model.train(mode=mode)
    return accuracy

def test_all(model, datasets, device, current_task, test_size=None, batch_size=512, verbose=True, base_log_dir=None, final_ts=None):
    """
    Evaluates the model on a list of datasets corresponding to each task.
    Logs confusion matrices and classification reports for the current task and cumulative tasks.
    """
    from sklearn.metrics import confusion_matrix, classification_report
    n_tasks = len(datasets)
    precs = []
    for i in range(n_tasks):
        acc = test_acc(model, datasets[i], device, task_id=None,
                       test_size=test_size, batch_size=batch_size)
        precs.append(acc)
    ave_so_far = sum(precs[:current_task]) / current_task if current_task > 0 and len(precs) >= current_task else 0.0
    if len(precs) >= current_task and current_task > 0:
        ave_this_task = precs[current_task-1]
    elif precs:
        ave_this_task = precs[-1]
    else:
        print("[Warning] precs is empty, defaulting accuracy to 0.")
        ave_this_task = 0.0
    if verbose:
        print(f' => Ave accuracy (this task):    {ave_this_task:.3f}')
        print(f' => Ave accuracy (tasks so far): {ave_so_far:.3f}')
    
    # Log current task evaluation metrics.
    current_dataset = datasets[current_task-1]
    current_loader = DataLoader(current_dataset, batch_size=batch_size, shuffle=False)
    preds, labels = evaluate_model(model, current_loader, device)
    cm_current = confusion_matrix(labels, preds)
    report_current = classification_report(labels, preds, digits=3)
    fig_cm_current = plot_confusion_matrix(cm_current, class_names=[str(cls) for cls in np.unique(labels)])
    cm_tensor_current = figure_to_tensor(fig_cm_current)
    
    # Log cumulative evaluation metrics (for all tasks up to current).
    from torch.utils.data import ConcatDataset
    cumulative_dataset = ConcatDataset(datasets[:current_task])
    cumulative_loader = DataLoader(cumulative_dataset, batch_size=batch_size, shuffle=False)
    preds_all, labels_all = evaluate_model(model, cumulative_loader, device)
    cm_all = confusion_matrix(labels_all, preds_all)
    report_all = classification_report(labels_all, preds_all, digits=3)
    fig_cm_all = plot_confusion_matrix(cm_all, class_names=[str(cls) for cls in np.unique(labels_all)])
    cm_tensor_all = figure_to_tensor(fig_cm_all)
    
    # Log to TensorBoard if logging directory and timestamp provided.
    if base_log_dir is not None and final_ts is not None:
        current_log_dir = os.path.join(base_log_dir, f"{final_ts}_finalTaskVal")
        writer_current = SummaryWriter(log_dir=current_log_dir)
        writer_current.add_image("Confusion Matrix", cm_tensor_current, 0)
        writer_current.add_text("Classification Report", report_current, 0)
        writer_current.close()
        
        cumulative_log_dir = os.path.join(base_log_dir, f"{final_ts}_finalAllVal")
        writer_cumulative = SummaryWriter(log_dir=cumulative_log_dir)
        writer_cumulative.add_image("Confusion Matrix", cm_tensor_all, 0)
        writer_cumulative.add_text("Classification Report", report_all, 0)
        writer_cumulative.close()
    
    return precs

# ------------------------------------------------------------------------------
# Continual Training Pipeline
def continual_training_pipeline(config, model_factory, device):
    """
    Main pipeline that iterates through tasks, performs hyperparameter tuning, trains the model,
    [Optionally updates the model output layer], logs metrics, and evaluates performance.
    
    Note: For now, the model's output is fixed at 8 classes.
          The code to update (expand) the final output layer is commented out.
    """
    # Generate a final timestamp in the desired format "HHMMddMM" (e.g., "15300506")
    final_ts = config["experiment"].get("timestamp", time.strftime("%H%M%d%m"))
    base_log_dir = config["logging"]["base_log_dir"]
    method = config["experiment"].get("continual_method", "ewc")
    print(f"[INFO] Starting continual learning training pipeline with method: {method}")
    print("[INFO] Configuration being used:")
    print(yaml.dump(config, default_flow_style=False))
    
    task_list = config["experiment"]["task_list"]
    model = None
    test_datasets = []
    
    # Loop through each task in the task list
    for task_idx, local_classes in enumerate(task_list):
        # Compute global classes and mapping for this task
        global_classes = [cls for sublist in task_list[:task_idx+1] for cls in sublist]
        global_num_classes = len(global_classes)  # Should remain 8 for now
        global_class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}
        target_transform = lambda y: global_class_to_idx[int(y)]
        
        # Load training and test datasets for the current global classes
        global_train_ds = WaferMapDataset(
            file_path=config["dataset"]["path"],
            split="train",
            oversample=False,
            target_dim=(224, 224),
            task_classes=global_classes
        )
        train_dataset = SubDataset(global_train_ds, local_classes, target_transform=target_transform)
        global_test_ds = WaferMapDataset(
            file_path=config["dataset"]["path"],
            split="test",
            oversample=False,
            target_dim=(224, 224),
            task_classes=global_classes
        )
        test_dataset = SubDataset(global_test_ds, local_classes, target_transform=target_transform)
        test_datasets.append(test_dataset)
        
        # Hyperparameter tuning for the current task
        best_lr, best_wd = tune_hyperparameters_for_task(
            config, train_dataset, model_factory, device,
            num_epochs=config["experiment"]["num_epochs"],
            num_trials=config["experiment"]["num_trials"],
            num_classes=global_num_classes,
            task_list=task_list,
            task_idx=task_idx,
            local_classes=local_classes
        )
        
        print(f"[INFO] Best hyperparameters for Task {task_idx}: lr={best_lr}, weight_decay={best_wd}")
        
        # Initialize or update model based on task index
        if task_idx == 0:
            model, criterion, optimizer = model_factory(best_lr, best_wd, global_classes, device)
        else:
            # For now, we want to keep the output fixed at 8 classes.
            # Commenting out the update_model_output call:
            old_num_classes = sum(len(task_list[j]) for j in range(task_idx))
            new_classes_count = global_num_classes - old_num_classes
            model = update_model_output(model, new_classes_count=new_classes_count, device=device)
            optimizer = torch.optim.SGD(model.parameters(), lr=best_lr, weight_decay=best_wd)
            optimizer = torch.optim.SGD(model.parameters(), lr=best_lr, weight_decay=best_wd)
        
        print_model_info(model, task_idx)
        
        # Create a SummaryWriter for final training logs for this task.
        # Final training log directory name: "{final_ts}_Task{task_idx}_finalTrain"
        final_train_log_dir = os.path.join(base_log_dir, f"{final_ts}_Task{task_idx}_finalTrain")
        writer = SummaryWriter(log_dir=final_train_log_dir)
        
        # Set up DataLoaders for training and validation.
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Lists to store epoch metrics.
        train_losses = []
        val_losses = []
        
        # Train using either EWC or baseline method.
        if config["experiment"].get("continual_method") == "ewc" and task_idx > 0:
            num_batches = len(train_loader)
            for epoch in range(config["experiment"]["num_epochs"]):
                epoch_losses = train_ewc(
                    model, train_dataset,
                    iters=num_batches,
                    lr=best_lr, batch_size=64,
                    device=device, current_task=task_idx,
                    ewc_lambda=config["experiment"].get("ewc_lambda", 100.0),
                    task_classes=global_classes
                )
                train_loss = np.mean(epoch_losses)
                train_losses.append(train_loss)
                val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
                val_losses.append(val_loss)
                writer.add_scalar(f"Train/Loss_Task{task_idx}", train_loss, epoch)
                writer.add_scalar(f"Validation/Loss_Task{task_idx}", val_loss, epoch)
                print(f"[INFO] Task {task_idx} (EWC) Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        else:
            for epoch in range(config["experiment"]["num_epochs"]):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                train_losses.append(train_loss)
                val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
                val_losses.append(val_loss)
                writer.add_scalar(f"Train/Loss_Task{task_idx}", train_loss, epoch)
                writer.add_scalar(f"Validation/Loss_Task{task_idx}", val_loss, epoch)
                print(f"[INFO] Task {task_idx} Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        writer.close()
        print(f"[INFO] Finished Task {task_idx}")
        
        # Update Fisher information if using EWC.
        if config["experiment"].get("continual_method") == "ewc":
            estimate_fisher(
                model, 
                train_dataset, 
                n_samples=500,       
                device=device, 
                ewc_gamma=0.9,        
                batch_size=8
            )
        # Save model checkpoint after each task.
        save_model_checkpoint(model, config, task_idx=task_idx, timestamp=final_ts)
        print(f"[INFO] Evaluating performance for tasks 1 to {task_idx+1}...")
        
        # Log evaluation metrics via test_all.
        test_all(model, test_datasets, device, current_task=task_idx+1, test_size=None, verbose=True,
                 base_log_dir=base_log_dir, final_ts=final_ts)
    
    print("[INFO] Continual learning training completed.")
    return model

# ------------------------------------------------------------------------------
# Main function to parse arguments, load configuration, and start the pipeline.
def main():
    args = parse_args()
    config = load_config(args.config)
    print("[INFO] Loaded configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Set seed for reproducibility if configured.
    if config["experiment"].get("reproducibility", False):
        seed = config["experiment"].get("seed", 42)
        from utils import set_seed
        set_seed(seed)
        print(f"[INFO] Reproducibility enabled. Seed set to {seed}")
    
    # Choose model factory based on configuration.
    if config["model"]["type"] == "resnet50":
        from Models.Wafer_resnet_model import create_resnet_model
        model_factory = create_resnet_model
    elif config["model"]["type"] == "simplenn":
        from Models.Wafer_simple_model import create_simple_model
        model_factory = create_simple_model
    
    device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"[INFO] Using device: {device}")
    
    final_model = continual_training_pipeline(config, model_factory, device)
    return final_model

if __name__ == '__main__':
    main()
