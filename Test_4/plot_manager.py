import matplotlib.pyplot as plt
import random

class PlotManager:
    @staticmethod
    def plot_sample_images(dataloader, title="Sample Images"):
        samples = []
        labels = []
        for inputs, targets in dataloader:
            for input, target in zip(inputs, targets):
                samples.append(input)
                labels.append(target)
        
        random_indices = random.sample(range(len(samples)), 30)
        random_samples = [samples[i] for i in random_indices]
        random_labels = [labels[i] for i in random_indices]

        fig, axes = plt.subplots(3, 10, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(random_samples[i][0], cmap='gray')
            ax.set_title(f'Label: {random_labels[i].item()}')
            ax.axis('off')
        plt.show()

    @staticmethod
    def plot_results(results):
        tasks = list(results.keys())
        accuracies = list(results.values())
        plt.plot(tasks, accuracies, marker='o')
        plt.xlabel('Tasks')
        plt.ylabel('Accuracy')
        plt.title('Task Accuracy Over Time')
        plt.show()

    @staticmethod
    def plot_train_results(train_losses, val_accuracies, title='Training and Validation'):
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'r', label='Training loss')
        plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
        plt.title(f'{title} - Loss and Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_avg_losses(train_losses_list, val_accuracies_list, title='Average Training and Validation Loss'):
        avg_train_losses = [sum(epoch_losses)/len(epoch_losses) for epoch_losses in train_losses_list]
        avg_val_accuracies = [sum(epoch_accuracies)/len(epoch_accuracies) for epoch_accuracies in val_accuracies_list]
        epochs = range(1, len(avg_train_losses) + 1)
        plt.plot(epochs, avg_train_losses, 'r', label='Average Training loss')
        plt.plot(epochs, avg_val_accuracies, 'b', label='Average Validation accuracy')
        plt.title(f'{title} - Average Loss and Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.show()
