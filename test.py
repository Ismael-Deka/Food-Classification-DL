import pickle as pk
import torch
import sys

print("Loading test dataset...")

try:
    with open('pickle/test.pkl', 'rb') as handle:
        test_dataset = pk.load(handle)
        test_loader = pk.load(handle)
        batch_size = pk.load(handle)
        num_epochs = pk.load(handle)
except FileNotFoundError:
    print("Test dataset not found. Please run train.py before running test.")
    sys.exit()

print("Loading fine-tuned model...")

try:
    model = torch.load("model/resnet50_food_classification_trained.pth")
except FileNotFoundError:
    print("Fine-tuned model not found. Please run train.py before running test.")
    sys.exit()