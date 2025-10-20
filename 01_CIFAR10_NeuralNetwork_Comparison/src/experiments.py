import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import pandas as pd
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt

def run_experiments(x_train, y_train, x_test, y_test):
    """
    Main function to run all experiments for both MLP and CNN models
    """
    print("🚀 Starting all experiments...")

    # Set device - Using MPS for Apple Silicon as mentioned in assignment tips
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert numpy arrays to PyTorch tensors and create DataLoaders
    # Assignment requires using CIFAR-10 data with proper transforms (should be handled in data_loader)
    train_dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
    test_dataset = TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).long())

    # Batch size is part of training configuration we need to experiment with
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Import here to avoid circular imports
    from train import train_model
    from models import create_mlp_model, create_cnn_model

    # Requirement: Train both MLP (≥2 hidden layers) and CNN (≥2 hidden layers)
    print("\n📊 Training MLP model (≥2 hidden layers)...")
    mlp_model = create_mlp_model()
    mlp_model.to(device)
    # Requirement: Must use cross-entropy loss (handled in train_model)
    mlp_history = train_model(mlp_model, train_loader, val_loader, device=device, model_name='mlp')

    print("\n📊 Training CNN model (≥2 hidden layers)...")
    cnn_model = create_cnn_model()
    cnn_model.to(device)
    cnn_history = train_model(cnn_model, train_loader, val_loader, device=device, model_name='cnn')

    # Requirement: ConvNet should outperform MLP in prediction performance
    # This will be verified during training and in the report

    # Requirement: Additional analysis experiment
    print("\n🔬 Running additional experiment: Learning Rate Analysis...")
    lr_results = learning_rate_experiment(
        create_mlp_model,
        train_loader,
        val_loader,
        device=device
    )

    # Requirement: Save results as data files (CSV supported by Pandas)
    os.makedirs('results', exist_ok=True)
    lr_results.to_csv('results/learning_rate_experiment.csv', index=False)

    # Also save training histories for learning curves
    pd.DataFrame(mlp_history).to_csv('results/mlp_training_history.csv', index=False)
    pd.DataFrame(cnn_history).to_csv('results/cnn_training_history.csv', index=False)

    print("✅ All experiments completed! Results saved to results/ folder")
    return lr_results


def learning_rate_experiment(model_class, train_loader, val_loader,
                             learning_rates=[0.001, 0.01, 0.1, 0.5],
                             num_epochs=3, device='cpu'):
    """
    Additional experiment: How do different learning rates affect model performance?
    This meets the assignment requirement for an extra analysis.
    """
    results = []

    print(f"Testing {len(learning_rates)} learning rates over {num_epochs} epochs each")

    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")

        # Initialize model with cross-entropy loss as required
        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()  # REQUIRED: Cross-entropy loss
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # Using SGD as mentioned in assignment examples

        # Train for a few epochs
        start_time = time.time()
        epoch_losses = []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            batch_count = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            avg_epoch_loss = running_loss / batch_count
            epoch_losses.append(avg_epoch_loss)

        training_time = time.time() - start_time

        # Evaluate on validation set (using test partition as validation per assignment)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        results.append({
            'learning_rate': lr,
            'training_time': training_time,
            'final_train_loss': epoch_losses[-1],
            'final_val_loss': val_loss,
            'final_val_accuracy': val_acc,
            'avg_epoch_loss': sum(epoch_losses) / len(epoch_losses)
        })

        print(f"   Val Accuracy: {val_acc:.2f}%, Time: {training_time:.2f}s")

    # Requirement: Return data for plotting and report
    return pd.DataFrame(results)


def evaluate_model(model, data_loader, criterion, device='cpu'):
    """
    Evaluate model on validation data (using test partition as validation per assignment requirement)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            batch_count += 1

    loss = running_loss / batch_count
    accuracy = 100. * correct / total

    return loss, accuracy