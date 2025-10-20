import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def calculate_mean_std(loader):
    """
    Calculate mean and standard deviation of a dataset from DataLoader
    for normalization purposes.
    """
    # Initialize variables
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        # Ensure data is float tensor
        data = data.float()

        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    # Calculate mean and standard deviation
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def plot_training_history(history, model_name, save_path=None):
    """
    Plot training history (loss and accuracy curves)
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    axes[0].plot(history['epoch'], history['train_loss'], label='Training Loss')
    axes[0].plot(history['epoch'], history['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot accuracy
    axes[1].plot(history['epoch'], history['train_acc'], label='Training Accuracy')
    axes[1].plot(history['epoch'], history['val_acc'], label='Validation Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path=None):
    """
    Plot confusion matrix
    """

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def get_predictions(model, data_loader, device='cpu'):
    """
    Get predictions from a model
    """

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def save_results(history, filename):
    """
    Save training results to CSV file
    """

    df = pd.DataFrame(history)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")