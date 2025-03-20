import os
import yaml
import random
import numpy as np
import torch
from datetime import datetime as time
import io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report
from Wafer_data_dataset_resize import WaferMapDataset
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from torch.utils.data import Subset
import PIL.Image as Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning Training Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    return parser.parse_args()

def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file and returns a dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["dataset"]["path"] = os.path.expanduser(config["dataset"]["path"])
    return config

def set_seed(seed=42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def update_model_output(model, new_classes_count, device):
    old_fc = model.fc
    in_features = old_fc.in_features
    current_out = old_fc.out_features
    new_out = current_out + new_classes_count
    new_fc = torch.nn.Linear(in_features, new_out).to(device)

    # Critical step explicitly copy old weights clearly:
    new_fc.weight.data[:current_out] = old_fc.weight.data
    new_fc.bias.data[:current_out] = old_fc.bias.data

    model.fc = new_fc
    return model

def save_model_checkpoint(model, config, task_idx=None, timestamp=None):
    """
    Save the model's state dictionary in an organized directory structure.
    """
    base_dir = config["experiment"].get("checkpoint_base_dir", "model_checkpoints")
    method = config["experiment"].get("continual_method", "baseline")
    if timestamp is None:
        # Use time.now() to get the current datetime and then format it.
        timestamp = time.now().strftime("%Y%m%d_%H%M%S")
    
    if task_idx is not None:
        folder = os.path.join(base_dir, method, timestamp, f"task{task_idx}")
        os.makedirs(folder, exist_ok=True)
        filename = f"model_{timestamp}.pth"
    else:
        folder = os.path.join(base_dir, method, timestamp, "final")
        os.makedirs(folder, exist_ok=True)
        filename = config["experiment"].get("final_model_filename", "final_model.pth")
    
    save_path = os.path.join(folder, filename)
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model checkpoint saved to {save_path}")
    return save_path


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # Debug clearly:
        print(f"Outputs shape: {outputs.shape}, Labels: {labels}")

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
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title="Confusion Matrix",
           ylabel="True label",
           xlabel="Predicted label")
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
    Plots training and validation curves for a given metric.
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
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    buf.close()
    return torch.tensor(img)

def evaluate_cumulative_tasks(config, model, device, task_list):
    cumulative_datasets = []

    print("\n[INFO] Starting cumulative evaluation...")

    for task_idx, classes in enumerate(task_list):
        print(f"\n[Evaluation] Task {task_idx} (classes: {classes})")

        # Load current task test dataset explicitly
        current_test_dataset = WaferMapDataset(
            file_path=config["dataset"]["path"],
            split="test",
            oversample=False,
            target_dim=(224, 224),
            task_classes=classes
        )
        current_loader = DataLoader(current_test_dataset, batch_size=64, shuffle=False)

        # Evaluate on the current task dataset
        preds, labels = evaluate_model(model, current_loader, device)
        current_acc = (preds == labels).mean()
        current_report = classification_report(labels, preds, digits=3)
        print(f"Current Task {task_idx} Accuracy: {current_acc:.4f}")
        print(current_report)

        # Add current dataset to cumulative list
        cumulative_datasets.append(current_test_dataset)

        # For the first task, cumulative test is same as current test
        if task_idx == 0:
            continue  # Skip cumulative test since it's identical to current

        # Combine datasets for cumulative evaluation
        cumulative_dataset = ConcatDataset(cumulative_datasets)
        cumulative_loader = DataLoader(cumulative_dataset, batch_size=64, shuffle=False)

        # Evaluate cumulatively (tasks 0..task_idx)
        cum_preds, cum_labels = evaluate_model(model, cumulative_loader, device)
        cum_acc = (cum_preds == cum_labels).mean()
        cum_report = classification_report(cum_labels, cum_preds, digits=3)

        print(f"Cumulative Accuracy (tasks 0 to {task_idx}): {cum_acc:.4f}")
        print(cum_report)

def apply_mask(outputs, task_id):
    """
    Applies a mask to the outputs so that only the entries corresponding 
    to the current task are retained.
    """
    mask = torch.zeros_like(outputs)
    start = 2 * task_id
    end = 2 * (task_id + 1)
    mask[:, start:end] = 1.0
    return outputs * mask

def plot_training_curve(train_values, val_values, metric_name="Loss"):
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



def robust_stratified_split(dataset, n_splits=3, oversample_small_classes=True, seed=42):
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