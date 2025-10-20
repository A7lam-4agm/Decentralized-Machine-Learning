import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import pandas as pd
from tqdm import tqdm
import os


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=25, device='cpu', model_name='model',
                checkpoint_dir='./results/model_checkpoints'):
    """
    Train a model and return training history
    """

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)

    # Track training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{checkpoint_dir}/{model_name}_best.pth')

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'{checkpoint_dir}/{model_name}_epoch_{epoch + 1}.pth')

    # Save final model
    torch.save(model.state_dict(), f'{checkpoint_dir}/{model_name}_final.pth')

    return history


def evaluate_model(model, test_loader, criterion, device='cpu'):
    """
    Evaluate model on test/validation data
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = running_loss / len(test_loader)
    acc = 100. * correct / total

    return loss, acc


def get_device():
    """
    Get the available device (GPU if available, else CPU)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')