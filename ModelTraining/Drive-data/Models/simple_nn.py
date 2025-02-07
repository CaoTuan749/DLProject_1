# simple_nn.py
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def create_simple_model(lr, weight_decay, input_dim, hidden_dim, output_dim):
    """
    Factory function to create an instance of the Model.
    
    Args:
        lr (float): Learning rate (not used here but passed to conform to the expected signature).
        weight_decay (float): Weight decay (not used here but passed to conform to the expected signature).
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in the first linear layer.
        output_dim (int): Number of output units.
    
    Returns:
        Model: An instance of the Model class.
    """
    return Model(input_dim, hidden_dim, output_dim)
