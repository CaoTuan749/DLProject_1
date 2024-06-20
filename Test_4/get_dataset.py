import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter

class GetDataset:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=self.transform)
        self.mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=self.transform)

    def get_task_dataset(self, task_labels):
        indices = [i for i, label in enumerate(self.mnist_train.targets) if label in task_labels]
        subset = Subset(self.mnist_train, indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=True)
        self.print_loader_stats(dataloader, "Training", task_labels, subset)
        return dataloader

    def get_test_dataset(self, task_labels):
        indices = [i for i, label in enumerate(self.mnist_test.targets) if label in task_labels]
        subset = Subset(self.mnist_test, indices)
        dataloader = DataLoader(subset, batch_size=1000, shuffle=False)
        self.print_loader_stats(dataloader, "Testing", task_labels, subset)
        return dataloader

    def print_loader_stats(self, dataloader, phase, task_labels, subset):
        num_batches = len(dataloader)
        batch_size = dataloader.batch_size
        num_samples = len(dataloader.dataset)
        
        # Accessing the targets directly from the subset indices
        targets = [subset.dataset.targets[i].item() for i in subset.indices]
        label_counter = Counter(targets)
        label_counts = {label: label_counter[label] for label in task_labels}
        
        print(f"{phase} Loader for labels {task_labels}:")
        print(f"  Number of batches: {num_batches}")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of samples: {num_samples}")
        print(f"  Samples per label: {label_counts}")

