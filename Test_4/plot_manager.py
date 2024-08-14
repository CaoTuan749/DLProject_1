# Standard libraries
import numpy as np
# Pytorch
import torch
# For visualization
from torchvision.utils import make_grid


class PlotManager:
    @staticmethod
    def multi_context_barplot(axis, accs, title=None):
        '''Generate barplot using the values in [accs].'''
        contexts = len(accs)
        axis.bar(range(contexts), accs, color='k')
        axis.set_ylabel('Testing Accuracy (%)')
        axis.set_xticks(range(contexts), [f'Context {i+1}' for i in range(contexts)])
        if title is not None:
            axis.set_title(title)

    @staticmethod
    def plot_examples(axis, dataset, context_id=None):
        '''Plot 25 examples from [dataset].'''
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=True)
        image_tensor, _ = next(iter(data_loader))
        image_grid = make_grid(image_tensor, nrow=5, pad_value=1) # pad_value=0 would give black borders
        axis.imshow(np.transpose(image_grid.numpy(), (1,2,0)))
        if context_id is not None:
            axis.set_title("Context {}".format(context_id+1))
        axis.axis('off')

    
    