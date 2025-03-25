#!/usr/bin/env python
"""
Simple Neural Network Module

This module defines a simple fully-connected neural network (SimpleNN) for image classification.
It processes images of size 224x224 with 3 channels and consists of three hidden layers followed by an output layer.
It also provides a factory function to create the model along with a CrossEntropyLoss and an Adam optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    """
    A simple fully-connected neural network for image classification.

    Architecture:
      - Flatten layer to convert input images (3 x 224 x 224) into a vector.
      - Three hidden fully-connected layers (each with 400 units) using ReLU activation.
      - An output layer with 'num_classes' units.
    """
    def __init__(self, num_classes: int):
        """
        Initializes the SimpleNN model.

        Args:
            num_classes (int): Number of output classes.
        """
        super(SimpleNN, self).__init__()
        # Flatten the input image (3 x 224 x 224) into a vector
        self.flatten = nn.Flatten()
        # Define the fully-connected layers
        self.fc1 = nn.Linear(3 * 224 * 224, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def create_simple_model(lr: float, weight_decay: float, num_classes: int, device: torch.device):
    """
    Factory function to create the SimpleNN model along with its loss function and optimizer.

    Args:
        lr (float): Learning rate for the Adam optimizer.
        weight_decay (float): Weight decay (L2 regularization factor) for the optimizer.
        num_classes (int): Number of output classes.
        device (torch.device): Device (CPU or GPU) on which the model is allocated.

    Returns:
        tuple: (model, criterion, optimizer)
            - model (SimpleNN): The SimpleNN model.
            - criterion (torch.nn.Module): CrossEntropyLoss for classification.
            - optimizer (torch.optim.Optimizer): Adam optimizer configured with given lr and weight_decay.
    """
    model = SimpleNN(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer
