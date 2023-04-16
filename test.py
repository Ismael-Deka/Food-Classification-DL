import torchvision.models as models
import pickle as pk
import torch
import sys

print("\nLoading test dataset...")

try:
    with open('pickle/test.pkl', 'rb') as handle:
        test_dataset = pk.load(handle)
        test_loader = pk.load(handle)
        batch_size = pk.load(handle)
        num_epochs = pk.load(handle)
        classes = pk.load(handle)
except FileNotFoundError:
    print("Test dataset not found. Please run train.py before running test.")
    sys.exit()

print("Loading fine-tuned model...")

try:
    model_dict = torch.load("model/resnet50_food_classification_trained.pth")
except FileNotFoundError:
    print("Fine-tuned model not found. Please run train.py before running test.")
    sys.exit()

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

num_classes = len(classes)