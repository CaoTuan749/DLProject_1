import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        # Adjust input features if your image size is different (this example assumes 64x64 images)
        self.fc1 = nn.Linear(224 * 224, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def create_simple_model(lr: float, weight_decay: float, num_classes: int, device):
    """
    Creates an instance of SimpleNN, along with a CrossEntropyLoss and an Adam optimizer.
    """
    model = SimpleNN(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer