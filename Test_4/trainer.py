import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):
        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in self.params.items():
            self._means[n] = p.data.clone()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            p.data.zero_()
            precision_matrices[n] = p.data.clone()

        self.model.eval()
        for input, _ in self.dataset:
            self.model.zero_grad()
            input = input.view(input.size(0), -1)
            output = self.model(input)
            label = output.max(1)[1]
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

class Trainer:
    def __init__(self, model, train_loader, test_loader, ewc=None, importance=1000, epochs = 5):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.ewc = ewc
        self.importance = importance
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_losses = []
        self.val_accuracies = []

    def train(self, use_ewc=False):
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Epoch Progress"):
            epoch_loss = 0
            for input, target in self.train_loader:
                input, target = input.view(input.size(0), -1), target
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = F.cross_entropy(output, target)
                # Print the loss before applying EWC
                #print(f"Epoch {epoch + 1}, Loss before EWC: {loss.item():.4f}")
                if use_ewc:
                    ewc_loss = self.importance * self.ewc.penalty(self.model)
                    # Print the EWC loss
                    #print(f"Epoch {epoch + 1}, EWC Loss: {ewc_loss.item():.4f}")
                    loss += ewc_loss
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            avg_loss = epoch_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for input, target in self.test_loader:
                input, target = input.view(input.size(0), -1), target
                output = self.model(input)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(self.test_loader.dataset)
        return accuracy

    def get_train_losses(self):
        return self.train_losses
