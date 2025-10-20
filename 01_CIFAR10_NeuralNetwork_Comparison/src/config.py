# Training configuration
BATCH_SIZE = 64
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Model configuration
MLP_HIDDEN_SIZES = [512, 256]
CNN_DROPOUT_RATE = 0.3

# Experiment configuration
EXPERIMENT_LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]
EXPERIMENT_BATCH_SIZES = [32, 64, 128, 256]
EXPERIMENT_EPOCHS = 10

# Path configuration
DATA_DIR = './data'
RESULTS_DIR = './results'
MODEL_CHECKPOINT_DIR = './results/model_checkpoints'
PLOT_DIR = './results/plots'

# CIFAR class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]