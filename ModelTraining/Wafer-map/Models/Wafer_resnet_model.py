import torch
import torch.nn as nn
from torchvision import models

def create_resnet_model(lr: float, weight_decay: float, num_classes: int, device: torch.device):
    """
    Creates a modified ResNet50 model for wafer map classification,
    along with a CrossEntropyLoss and an Adam optimizer.

    Parameters:
    - lr (float): Learning rate for the optimizer.
    - weight_decay (float): Weight decay for the optimizer.
    - num_classes (int): Number of output classes.
    - device (torch.device): The device on which to load the model.

    Returns:
    - model (nn.Module): The modified ResNet50 model.
    - criterion (nn.Module): The loss function (CrossEntropyLoss).
    - optimizer (torch.optim.Optimizer): The Adam optimizer.
    """
    print(f"[DEBUG] Creating ResNet50 with lr={lr}, weight_decay={weight_decay}")
    # Load a pretrained ResNet50 model
    model = models.resnet50(weights="IMAGENET1K_V1")
    # Replace the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Move the model to the specified device
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer