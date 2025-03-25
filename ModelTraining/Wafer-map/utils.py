#!/usr/bin/env python
"""
Utility functions for training pipelines.

This module provides functions for:
  - Parsing command-line arguments and loading configuration.
  - Setting seeds for reproducibility.
  - Updating model output layers (supports ResNet and SimpleNN).
  - Saving model checkpoints.
  - Training and validation routines.
  - Evaluation and plotting utilities (e.g., confusion matrix, training curves).
  - Robust stratified splitting of datasets.
"""

import os
import yaml
import random
import numpy as np
import torch
import time
from datetime import datetime as dt
import io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
import PIL.Image as Image
import argparse
from collections import Counter

# Import dataset class
from Wafer_data_dataset_resize import WaferMapDataset


def parse_args():
    """
    Parses command-line arguments for the continual learning training pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Continual Learning Training Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Expand user paths if necessary
    config["dataset"]["path"] = os.path.expanduser(config["dataset"]["path"])
    return config


def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_model_output(model, new_classes_count, device):
    """
    Updates the final classification layer of the model to accommodate new classes.
    Supports models with 'fc' (ResNet) or 'fc4' (SimpleNN) attributes.

    Args:
        model (torch.nn.Module): The model to update.
        new_classes_count (int): Number of new classes to add.
        device (torch.device): Device to allocate the updated layer.

    Returns:
        torch.nn.Module: Model with updated final layer.
    """
    # Determine which final layer attribute exists
    if hasattr(model, 'fc'):
        old_fc = model.fc
    elif hasattr(model, 'fc4'):
        old_fc = model.fc4
    else:
        raise AttributeError("Model does not have a known final layer (neither 'fc' nor 'fc4').")
    
    in_features = old_fc.in_features
    current_out = old_fc.out_features
    new_out = current_out + new_classes_count

    # Create a new final layer with updated output size
    new_fc = torch.nn.Linear(in_features, new_out).to(device)
    # Copy existing weights and biases for the old classes
    new_fc.weight.data[:current_out] = old_fc.weight.data
    new_fc.bias.data[:current_out] = old_fc.bias.data

    # Reassign the new layer to the model
    if hasattr(model, 'fc'):
        model.fc = new_fc
    elif hasattr(model, 'fc4'):
        model.fc4 = new_fc

    return model


def save_model_checkpoint(model, config, task_idx=None, timestamp=None):
    """
    Save the model's state dictionary in an organized directory structure with unique filenames.
    """
    base_dir = config["experiment"].get("checkpoint_base_dir", "model_checkpoints")
    method = config["experiment"].get("continual_method", "baseline")
    if timestamp is None:
        # Use the current datetime to format a timestamp.
        timestamp = time.now().strftime("%Y%m%d_%H%M%S")
    
    if task_idx is not None:
        folder = os.path.join(base_dir, method, timestamp, f"task{task_idx}")
        os.makedirs(folder, exist_ok=True)
        # Unique filename including task index and timestamp.
        filename = f"model_task{task_idx}_{timestamp}.pth"
    else:
        folder = os.path.join(base_dir, method, timestamp, "final")
        os.makedirs(folder, exist_ok=True)
        filename = config["experiment"].get("final_model_filename", "final_model.pth")
    
    save_path = os.path.join(folder, filename)
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model checkpoint saved to {save_path}")
    return save_path


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: Average loss and training accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
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
    """
    Validates the model for one epoch.

    Args:
        model (torch.nn.Module): The model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (callable): Loss function.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: Average loss and validation accuracy for the epoch.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(dataloader), correct / total


def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on a given dataset.

    Args:
        model (torch.nn.Module): The model.
        dataloader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: Numpy arrays of predictions and ground-truth labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names):
    """
    Plots the confusion matrix using matplotlib.

    Args:
        cm (ndarray): Confusion matrix.
        class_names (list): List of class names.

    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_training_curve(train_values, val_values, metric_name="Loss"):
    """
    Plots the training and validation curves for a given metric.

    Args:
        train_values (list): List of training metric values.
        val_values (list): List of validation metric values.
        metric_name (str): Name of the metric (default "Loss").

    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(train_values) + 1)
    ax.plot(epochs, train_values, label=f"Train {metric_name}")
    ax.plot(epochs, val_values, label=f"Validation {metric_name}")
    ax.set_title(f"Training and Validation {metric_name} Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.legend()
    fig.tight_layout()
    return fig


def figure_to_tensor(figure):
    """
    Converts a matplotlib figure to a PyTorch tensor.

    Args:
        figure (matplotlib.figure.Figure): Figure to convert.

    Returns:
        torch.Tensor: Tensor representation of the figure.
    """
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    buf.close()
    return torch.tensor(img)


def evaluate_cumulative_tasks(config, model, device, task_list):
    """
    Evaluates the model on cumulative tasks, printing accuracy and classification reports.

    Args:
        config (dict): Configuration dictionary.
        model (torch.nn.Module): The model.
        device (torch.device): Device for computation.
        task_list (list): List of task class lists.
    """
    cumulative_datasets = []
    print("\n[INFO] Starting cumulative evaluation...")

    for task_idx, classes in enumerate(task_list):
        print(f"\n[Evaluation] Task {task_idx} (classes: {classes})")
        # Load current task test dataset
        current_test_dataset = WaferMapDataset(
            file_path=config["dataset"]["path"],
            split="test",
            oversample=False,
            target_dim=(224, 224),
            task_classes=classes
        )
        current_loader = DataLoader(current_test_dataset, batch_size=64, shuffle=False)
        # Evaluate on current task
        preds, labels = evaluate_model(model, current_loader, device)
        current_acc = (preds == labels).mean()
        current_report = classification_report(labels, preds, digits=3)
        print(f"Current Task {task_idx} Accuracy: {current_acc:.4f}")
        print(current_report)
        cumulative_datasets.append(current_test_dataset)
        # Skip cumulative evaluation for the first task
        if task_idx == 0:
            continue
        # Combine datasets for cumulative evaluation
        cumulative_dataset = ConcatDataset(cumulative_datasets)
        cumulative_loader = DataLoader(cumulative_dataset, batch_size=64, shuffle=False)
        cum_preds, cum_labels = evaluate_model(model, cumulative_loader, device)
        cum_acc = (cum_preds == cum_labels).mean()
        cum_report = classification_report(cum_labels, cum_preds, digits=3)
        print(f"Cumulative Accuracy (tasks 0 to {task_idx}): {cum_acc:.4f}")
        print(cum_report)


def apply_mask(outputs, task_id):
    """
    Applies a mask to outputs so that only entries corresponding to the current task are retained.

    Args:
        outputs (torch.Tensor): Model outputs.
        task_id (int): Current task ID.

    Returns:
        torch.Tensor: Masked outputs.
    """
    mask = torch.zeros_like(outputs)
    start = 2 * task_id
    end = 2 * (task_id + 1)
    mask[:, start:end] = 1.0
    return outputs * mask


def print_model_info(model, task_idx):
    """
    Prints information about the model's parameters and output layer.

    Args:
        model (torch.nn.Module): The model.
        task_idx (int): Current task index.
    """
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


def robust_stratified_split(dataset, n_splits=3, oversample_small_classes=True, seed=42):
    """
    Splits a dataset into stratified folds, with optional oversampling for small classes.

    Args:
        dataset: Dataset with a 'y' attribute for labels.
        n_splits (int): Number of splits.
        oversample_small_classes (bool): Whether to oversample small classes.
        seed (int): Random seed.

    Returns:
        list: List of dictionaries with 'train' and 'val' subsets.
    """
    labels = np.array(dataset.y) if hasattr(dataset, 'y') else np.array([dataset.dataset.y[i] for i in dataset.indices])
    class_counts = Counter(labels)
    min_class_count = min(class_counts.values())
    # Adjust number of splits safely
    safe_n_splits = min(n_splits, min_class_count)
    safe_n_splits = max(safe_n_splits, 2)
    indices = np.arange(len(labels)).reshape(-1, 1)

    if oversample_small_classes and min_class_count < safe_n_splits:
        ros = RandomOverSampler(random_state=seed)
        indices_resampled, labels_resampled = ros.fit_resample(indices, labels)
        indices_resampled = indices_resampled.flatten()
    else:
        indices_resampled, labels_resampled = indices.flatten(), labels

    skf = StratifiedKFold(n_splits=safe_n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(indices_resampled, labels_resampled):
        splits.append({
            'train': Subset(dataset, indices_resampled[train_idx]),
            'val': Subset(dataset, indices_resampled[val_idx])
        })
    return splits
