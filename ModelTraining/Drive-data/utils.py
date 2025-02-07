# utils.py
import os
import yaml
from torch.utils.tensorboard import SummaryWriter

def load_config(config_path="config.yaml"):
    """
    Loads YAML configuration from the given file path and returns a dictionary.
    Expands the tilde in the dataset path if present.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "dataset" in config and "path" in config["dataset"]:
        config["dataset"]["path"] = os.path.expanduser(config["dataset"]["path"])
    return config

def get_tensorboard_writer(log_dir):
    """
    Returns a TensorBoard SummaryWriter instance.
    """
    writer = SummaryWriter(log_dir=log_dir)
    return writer
