import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable hidden layers
    """

    def __init__(self, input_size=32 * 32 * 3, hidden_sizes=[512, 256], num_classes=10, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.input_size = input_size

        # Create layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, self.input_size)
        return self.network(x)


class CNN(nn.Module):
    """
    Convolutional Neural Network for image classification
    """

    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(CNN, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def create_mlp_model(input_size=32*32*3, hidden_size=512, output_size=10):
    """
    Creates and returns an MLP model instance.
    """
    model = MLP(input_size, hidden_size, output_size)
    return model

def create_cnn_model():
    """
    Creates and returns a CNN model instance.
    """
    model = CNN()
    return model

