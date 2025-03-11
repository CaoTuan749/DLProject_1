import torch
import torch.nn.functional as F
import tqdm
from utils import apply_mask

def train_ewc(model, dataset, iters, lr, batch_size, current_task=None, ewc_lambda=100., task_id=None, verbose=True):
    """
    Trains the model with EWC regularization.
    Parameters:
      - model: the network to be trained.
      - dataset: training dataset.
      - iters: total number of iterations.
      - lr: learning rate.
      - batch_size: batch size.
      - current_task: current task index (for EWC; tasks > 1 incur EWC loss).
      - ewc_lambda: regularization strength.
      - task_id: used for any task-specific output masking (if applicable).
      - verbose: whether to display a progress bar.
    Returns:
      - A list of loss values.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    model.train()
    iters_left = 1
    if verbose:
        progress_bar = tqdm.tqdm(range(1, iters + 1))
    losses = []
    for batch_index in range(1, iters + 1):
        iters_left -= 1
        if iters_left == 0:
            data_loader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                           shuffle=True, drop_last=True))
            iters_left = len(data_loader)
        x, y = next(data_loader)
        optimizer.zero_grad()
        y_hat = model(x)
        # Here, if you use task_id-specific masking, you should define apply_mask() separately.
        if task_id is not None:
                masked_outputs = apply_mask(y_hat, task_id)
                adjusted_labels = y - 2 * task_id          
                loss = torch.nn.functional.cross_entropy(
                        input=masked_outputs[:, 2*task_id:2*(task_id+1)], 
                        target=adjusted_labels)
                _, predicted = torch.max(masked_outputs[:, 2*task_id:2*(task_id+1)], 1)
                accuracy = (predicted == adjusted_labels).sum().item() * 100 / x.size(0)
        else:    
            # Here we assume the model has an attribute output_layers for its final layer.
            num_classes = model.output_layers.out_features if hasattr(model, 'output_layers') else y_hat.shape[1]
            adjusted_labels = y % num_classes
            loss = F.cross_entropy(y_hat, adjusted_labels)
            accuracy = (adjusted_labels == y_hat.max(1)[1]).sum().item() * 100 / x.size(0)

        if current_task is not None and current_task > 1:
            ewc_losses = []
            for n, p in model.named_parameters():
                n = n.replace('.', '__')
                mean = getattr(model, '{}_EWC_param_values'.format(n))
                fisher = getattr(model, '{}_EWC_estimated_fisher'.format(n))
                ewc_losses.append((fisher * (p - mean)**2).sum())
            ewc_loss = 0.5 * sum(ewc_losses)
            total_loss = loss + ewc_lambda * ewc_loss
        else:
            total_loss = loss

        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
        
        if verbose:
            progress_bar.set_description(
                '<EWC> | training loss: {loss:.3f} | training accuracy: {prec:.3f}%'
                .format(loss=total_loss.item(), prec=accuracy)
            )
            progress_bar.update(1)
    if verbose:
        progress_bar.close()
    return losses

def estimate_fisher(model, dataset, n_samples, ewc_gamma=1.):
    """
    Estimates the Fisher Information for each parameter of the model.
    Updates the model by registering buffers with:
      - the parameter values (mean) 
      - the estimated Fisher information.
    """
    est_fisher_info = {}
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        est_fisher_info[n] = p.detach().clone().zero_()
    mode = model.training
    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for index, (x, y) in enumerate(data_loader):
        if n_samples is not None and index > n_samples:
            break
        output = model(x)
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1)         
        for label_index in range(output.shape[1]):
            label = torch.LongTensor([label_index])
            negloglikelihood = F.cross_entropy(output, label)
            model.zero_grad()
            negloglikelihood.backward(retain_graph=True if (label_index+1) < output.shape[1] else False)
            for n, p in model.named_parameters():
                n = n.replace('.', '__')
                if p.grad is not None:
                    est_fisher_info[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)
    # Average the Fisher information
    est_fisher_info = {n: p / index for n, p in est_fisher_info.items()}
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        model.register_buffer('{}_EWC_param_values'.format(n), p.detach().clone())
        if hasattr(model, '{}_EWC_estimated_fisher'.format(n)):
            existing_values = getattr(model, '{}_EWC_estimated_fisher'.format(n))
            est_fisher_info[n] += ewc_gamma * existing_values
        model.register_buffer('{}_EWC_estimated_fisher'.format(n), est_fisher_info[n])
    model.train(mode=mode)
