#utils.py

import os
import yaml
import random
import numpy as np
import torch
from datetime import datetime as time
import io
import matplotlib.pyplot as plt

def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file and returns a dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Expand the tilde in the dataset path if present.
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
    # Enforce deterministic behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def update_model_output(model, new_classes_count, device):
    old_fc = model.fc
    in_features = old_fc.in_features
    current_out = old_fc.out_features
    new_out = current_out + new_classes_count
    new_fc = torch.nn.Linear(in_features, new_out).to(device)
    new_fc.weight.data[:current_out] = old_fc.weight.data.to(device)
    new_fc.bias.data[:current_out] = old_fc.bias.data.to(device)
    model.fc = new_fc
    return model


def save_model_checkpoint(model, config, task_idx=None, timestamp=None):
    """
    Save the model's state dictionary in an organized directory structure.
    Uses the base directory and final model filename from the config.
    
    If task_idx is provided, the model is saved under:
       checkpoint_base_dir/continual_method/timestamp/task{task_idx}/model_{timestamp}.pth
    Otherwise, it is saved as the final model in:
       checkpoint_base_dir/continual_method/timestamp/final/final_model_filename
    """
    base_dir = config["experiment"].get("checkpoint_base_dir", "model_checkpoints")
    method = config["experiment"].get("continual_method", "baseline")
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    
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

def figure_to_tensor(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    import PIL.Image as Image
    img = Image.open(buf)
    img = np.array(img).astype(np.float32) / 255.0  # normalize
    # Convert H x W x C to C x H x W
    img = np.transpose(img, (2, 0, 1))
    buf.close()
    return torch.tensor(img)

def evaluate_cumulative_tasks(config, model, device, task_list):
    """
    Evaluate the final model on each task's test set individually and on cumulative test sets.
    
    For each task t:
      - Evaluate the model on the test set for task t.
      - Combine test sets from task 0 to t and evaluate on the cumulative test set.
    """
    from Wafer_data_dataset_resize import WaferMapDataset  # Ensure dataset class is imported

    cumulative_X = []
    cumulative_y = []
    print("\n[INFO] Evaluating cumulative performance per task:")
    for t, classes in enumerate(task_list):
        # Load current task test set
        current_test_ds = WaferMapDataset(
            file_path=config["dataset"]["path"],
            split="test",
            oversample=False,
            target_dim=(224, 224),
            task_classes=classes
        )
        current_loader = DataLoader(current_test_ds, batch_size=64, shuffle=False)
        preds, labels = evaluate_model(model, current_loader, device)
        acc = (preds == labels).mean()
        print(f"Task {t} test set accuracy: {acc:.4f}")

        # Accumulate test data
        cumulative_X.append(current_test_ds.X)
        cumulative_y.append(current_test_ds.y)
        cum_X = np.concatenate(cumulative_X, axis=0)
        cum_y = np.concatenate(cumulative_y, axis=0)
        # Create a TensorDataset from cumulative data
        cum_tensor_X = torch.tensor(cum_X)
        cum_tensor_y = torch.tensor(cum_y)
        cum_dataset = torch.utils.data.TensorDataset(cum_tensor_X, cum_tensor_y)
        cum_loader = DataLoader(cum_dataset, batch_size=64, shuffle=False)
        cum_preds, cum_labels = evaluate_model(model, cum_loader, device)
        cum_acc = (cum_preds == cum_labels).mean()
        print(f"Cumulative test set accuracy (tasks 0..{t}): {cum_acc:.4f}")


def apply_mask(outputs, task_id):
    """
    Applies a mask to the outputs so that only the entries corresponding 
    to the current task are retained and all other entries are set to zero.
    
    Assumes each task uses 2 output units, so for a given task_id,
    only columns [2*task_id : 2*(task_id+1)] are active.
    
    Parameters:
      outputs (torch.Tensor): The raw model outputs of shape [batch_size, total_classes].
      task_id (int): The current task id.
      
    Returns:
      torch.Tensor: The masked outputs.
    """
    mask = torch.zeros_like(outputs)
    start = 2 * task_id
    end = 2 * (task_id + 1)
    # Set the relevant columns to one
    mask[:, start:end] = 1.0
    return outputs * mask
