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
from tqdm.auto import tqdm
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.exceptions import TrialPruned
import matplotlib.pyplot as plt
from collections import Counter


# Import shared utilities and dataset class.
from utils import train_one_epoch, validate_one_epoch, evaluate_model, save_model_checkpoint, update_model_output, apply_mask, print_model_info, parse_args, load_config
from Wafer_data_dataset_resize import WaferMapDataset

from torch.utils.data import Dataset
import torch

# ------------------------------------------------------------------------------
class SubDataset(Dataset):
    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        # Convert sub_labels to a tensor (we assume labels are integers)
        self.sub_labels = torch.tensor(sub_labels, dtype=torch.long)
        # Create a mask over the original dataset's labels (assumed to be in self.dataset.y)
        mask = torch.isin(torch.tensor(self.dataset.y), self.sub_labels)
        self.sub_indices = torch.nonzero(mask).squeeze().tolist()
        # Also, store the filtered labels as an attribute "y"
        # (if sub_indices is a single int, make it a list)
        if isinstance(self.sub_indices, int):
            self.y = [self.dataset.y[self.sub_indices]]
        else:
            self.y = np.array(self.dataset.y)[self.sub_indices].tolist()
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indices[index]]
        if self.target_transform:
            original_label = sample[1]
            target = self.target_transform(original_label)
            print(f"Original label: {original_label}, Transformed label: {target}")
            sample = (sample[0], target)
        return sample

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

def estimate_fisher(model, dataset, n_samples, device, ewc_gamma=1.0, batch_size=8):
    # Initialize Fisher Information storage
    est_fisher_info = {n.replace('.', '__'): torch.zeros_like(p, device=device) 
                       for n, p in model.named_parameters()}

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # Ensure n_samples does not exceed dataset size
    n_samples = min(n_samples, len(dataset))
    sampled_indices = np.random.choice(len(dataset), size=n_samples, replace=False)
    subset_dataset = Subset(dataset, sampled_indices)
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    total_processed = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        model.zero_grad()
        outputs = model(x)
        
        # Compute loss directly using true labels
        loss = criterion(outputs, y)
        loss.backward()

        batch_size_actual = x.size(0)
        total_processed += batch_size_actual

        # Accumulate squared gradients (Fisher Information)
        for n, p in model.named_parameters():
            param_name = n.replace('.', '__')
            if p.grad is not None:
                est_fisher_info[param_name] += (p.grad.detach() ** 2) * batch_size_actual

    # Normalize by total samples processed
    est_fisher_info = {n: p / total_processed for n, p in est_fisher_info.items()}

    # Store current parameter values and accumulated Fisher Information
    for n, p in model.named_parameters():
        param_name = n.replace('.', '__')
        
        # Store parameter values for EWC
        model.register_buffer(f'{param_name}_EWC_param_values', p.detach().clone())

        # Accumulate Fisher info if already exists from previous tasks
        if hasattr(model, f'{param_name}_EWC_estimated_fisher'):
            prev_fisher = getattr(model, f'{param_name}_EWC_estimated_fisher')
            est_fisher_info[param_name] += ewc_gamma * prev_fisher

        model.register_buffer(f'{param_name}_EWC_estimated_fisher', est_fisher_info[param_name])

    model.train()

def tune_hyperparameters_for_task(config, task_dataset, model_factory, device, num_epochs, num_trials, num_classes, task_list, task_idx, local_classes):

    from sklearn.model_selection import StratifiedKFold
    from tqdm.auto import tqdm
    import numpy as np
    import time

    def get_labels(ds):
        if hasattr(ds, 'y'):
            return np.array(ds.y)
        elif isinstance(ds, Subset):
            return np.array(ds.dataset.y)[ds.indices]
        else:
            raise AttributeError("Dataset does not have attribute 'y'")

    global_classes = [cls for sublist in task_list[:task_idx+1] for cls in sublist]
    global_class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}

    # Explicitly convert labels to integers before mapping:
    target_transform = lambda y: global_class_to_idx[int(y)]

    train_dataset_mapped = SubDataset(
        original_dataset=task_dataset.dataset,
        sub_labels=local_classes,
        target_transform=target_transform
    )

    all_labels = get_labels(train_dataset_mapped)
    all_indices = np.arange(len(train_dataset_mapped))

    skf = StratifiedKFold(n_splits=min(3, len(set(all_labels))), shuffle=True, random_state=42)

    trial_bar = tqdm(total=num_trials, desc=f"HPO Task {task_idx+1}")

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

        fold_bar = tqdm(enumerate(skf.split(all_indices, all_labels)), total=skf.get_n_splits(), desc="Folds", leave=False)

        for fold_idx, (train_idx, val_idx) in fold_bar:
            train_subset = Subset(train_dataset_mapped, train_idx)
            val_subset = Subset(train_dataset_mapped, val_idx)

            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

            model, criterion, optimizer = model_factory(
                trial_lr, trial_weight_decay, num_classes=len(global_classes), device=device
            )

            best_val_loss = float('inf')

            epoch_bar = tqdm(range(config["experiment"]["num_epochs"]), desc=f"Fold {fold_idx+1} Epochs", leave=False)

            for epoch in epoch_bar:
                epoch_start = time.time()

                train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, _ = validate_one_epoch(model, val_loader, criterion, device)

                epoch_duration = time.time() - epoch_start

                epoch_bar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'epoch_time': f'{epoch_duration:.1f}s'
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            fold_losses.append(best_val_loss)

        avg_loss = np.mean(fold_losses)
        trial_bar.update(1)
        trial_bar.set_postfix({'Avg Loss': f'{avg_loss:.4f}'})

        return avg_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)

    trial_bar.close()

    best_trial = study.best_trial
    print(f"[Debug] Best trial: lr={best_trial.params['lr']}, weight_decay={best_trial.params['weight_decay']}")

    return best_trial.params['lr'], best_trial.params['weight_decay']



# --- Updated Test Functions ---
def test_acc(model, dataset, device, task_id=None, test_size=2000, batch_size=512):
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

def test_all(model, datasets, device, current_task, test_size=None, batch_size=512, verbose=True):
    from tqdm.auto import tqdm
    n_tasks = len(datasets)
    precs = []

    task_bar = tqdm(range(n_tasks), desc="Tasks evaluation", leave=False)

    for i in task_bar:
        acc = test_acc(model, datasets[i], device, task_id=i if i == current_task-1 else None, 
                       test_size=test_size, batch_size=batch_size)
        precs.append(acc)

    # Safely handle empty or smaller precs list
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

    return precs

# ------------------------------------------------------------------------------
def continual_training_pipeline(config, model_factory, device):
    ts = config["experiment"].get("timestamp", "default")
    base_log_dir = config["experiment"].get("log_dir", "logs")
    method = config["experiment"].get("continual_method", "ewc")
    print(f"[INFO] Starting continual learning training pipeline with method: {method}")
    print("[INFO] Configuration being used:")
    print(yaml.dump(config, default_flow_style=False))
    task_list = config["experiment"]["task_list"]  # e.g., [[0,1], [2,3], [4,5]]
    model = None
    test_datasets = []  # List to store test datasets per task.
    for task_idx, local_classes in enumerate(task_list):
        global_classes = [cls for sublist in task_list[:task_idx+1] for cls in sublist]
        global_num_classes = len(global_classes)
        global_class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}

        # crucial fix: correct global indexing
        target_transform = lambda y: global_class_to_idx[int(y)]

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
        # Hyperparameter optimization.
        # Hyperparameter optimization (corrected call):
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
        # Update or initialize the model.
        if task_idx == 0:
            model, criterion, optimizer = model_factory(best_lr, best_wd, global_num_classes, device)
        else:
            old_num_classes = sum(len(task_list[j]) for j in range(task_idx))
            new_classes_count = global_num_classes - old_num_classes
            model = update_model_output(model, new_classes_count=new_classes_count, device=device)
            optimizer = torch.optim.SGD(model.parameters(), lr=best_lr, weight_decay=best_wd)
        print_model_info(model, task_idx)
        # TensorBoard logging.
        log_dir = os.path.join(config["logging"]["base_log_dir"], ts, f"task_{task_idx}")
        print(f"[INFO] TensorBoard log directory for Task {task_idx}: {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)
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
                print(f"[INFO] Task {task_idx} Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        writer.close()
        print(f"[INFO] Finished Task {task_idx}")
        if config["experiment"].get("continual_method") == "ewc":
            estimate_fisher(
                model, 
                train_dataset, 
                n_samples=500,       
                device=device, 
                ewc_gamma=0.9,        
                batch_size=8
            )
        save_model_checkpoint(model, config, task_idx=task_idx, timestamp=ts)
        # Evaluate performance for all tasks seen so far.
        print(f"[INFO] Evaluating performance for tasks 1 to {task_idx+1}...")
        test_all(model, test_datasets, device, current_task=task_idx+1, test_size=None, verbose=True)
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