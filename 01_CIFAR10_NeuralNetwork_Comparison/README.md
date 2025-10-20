# 📘 CS 595 – Project#1  
### CIFAR-10 Neural Network Comparison

This project trains and compares a **Multi-Layer Perceptron (MLP)** and a **Convolutional Neural Network (CNN)** on the **CIFAR-10** dataset using **PyTorch**.  
It also includes a small **learning-rate sweep** as the required “extra experiment” and produces all figures used in the IEEE-style report.
## Running Instructions

===============================================

## PREREQUISITES:
- Python 3.8+
- pip package manager

## QUICK START:
1. pip install -r requirements.txt
2. python main.py

## EXPECTED OUTPUT:
- Downloads CIFAR-10 dataset automatically
- Trains MLP (25 epochs) and CNN (25 epochs)
- Performs learning rate analysis experiment
- Generates results in /results/ folder

## OUTPUT FILES:
- mlp_results.csv, cnn_results.csv
- learning_rate_analysis.csv  
- Training plots (.pdf)
- Model checkpoints

## TRAINING TIME:
- CPU: ~30-60 minutes
- GPU: ~10-20 minutes (if available)

## OPTIONAL ARGUMENTS:
python main.py --epochs 10
python main.py --batch-size 32
python main.py --analyze-only

## NOTES:
- Models automatically use GPU if available
- First run downloads dataset (~200MB)
- Check /results/ folder for all outputs
