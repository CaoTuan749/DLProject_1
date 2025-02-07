import os
import yaml

def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file and returns a dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Expand the tilde in the dataset path if present.
    config["dataset"]["path"] = os.path.expanduser(config["dataset"]["path"])
    return config
