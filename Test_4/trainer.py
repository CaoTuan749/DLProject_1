import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):
        self.model = model
        self.dataset = dataset

        self._means = {}
        self._precision_matrices = self._diag_fisher()

        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        for n, p in self.params.items():
            self._means[n] = p.data.clone()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            precision_matrices[n] = p.detach().clone().zero_()

        mode = self.model.training
        self.model.eval()

        for index, (x,y) in enumerate(self.data_loader):
            self.model.zero_grad()
            output = self.model(x)

            with torch.no_grad():
                label_weights = F.softmax(output, dim=1)

            for label_index in range(output.shape[1]):
                label = torch.LongTensor([label_index])
                loss = F.cross_entropy(output, label)
                self.model.zero_grad()
                loss.backward()

                for n,p in self.model.named_parameters():
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        precision_matrices[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)

        precision_matrices = {n: p/index for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss



class Trainer:
    def __init__(self, model, lr, epochs, dataset, batch_size, current_task):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_task = current_task

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.model.train()
        epochs_left = 1
        progress_bar = tqdm.tqdm(range(1, self.epochs+1))

        for _ in range(1, self.epochs+1):
            epochs_left -= 1
            if epochs_left==0:
                data_loader = iter(torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                            shuffle=True, drop_last=True))
                epochs_left = len(data_loader)

            x, y = next(data_loader)
            optimizer.zero_grad()
            y_hat = self.model(x)
            loss = torch.nn.functional.cross_entropy(input=y_hat, target=y, reduction='mean')

            if self.current_task > 1:
                ewc = 
            else:
                total_loss = loss
            accuracy = (y == y_hat.max(1)[1]).sum().item()*100 / x.size(0)

            # Backpropagate errors
            loss.backward()
            optimizer.step()

            progress_bar.set_description(
            '<CLASSIFIER> | training loss: {loss:.3} | training accuracy: {prec:.3}% |'
                .format(loss=loss.item(), prec=accuracy)
            )
            progress_bar.update(1)
        progress_bar.close()       