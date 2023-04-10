import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch
import pickle as pk
import sys

from sklearn.model_selection import train_test_split

food_categories = ['burger', 'cake', 'cookie', 'fries', 'hotdog', 'pizza', 'salad', 'shrimp', 'steak', 'sushi']

labels = []

print("Loading dataset...")

for x in food_categories:
    labels.extend([x for _ in range(300)])


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize images
])

try:
    dataset = ImageFolder(root='dataset', transform=data_transforms)
except FileNotFoundError:
    print("Dataset not found.")
    sys.exit()

#Split original dataset and labels train(80% of original dataset) and test(20% of orginal dataset) sets with a random seed of 1
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=1) 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

#Saves our test set so we can use it later in test.py
with open('test.pickle', 'wb') as handle:
    pk.dump(X_test, handle)
    pk.dump(y_test, handle)

print("Loading pre-trained model...")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) #Loads pre-trained model ResNet50 with ImageNet weights

#Freezes all pre-trained layers of the original model so they don't get trained
for param in resnet50.parameters():
    param.requires_grad = False

#Our 10 food categories
num_classes = len(food_categories)

#Number of inputs in the final layer of resnet50
num_fc_inputs = resnet50.fc.in_features

#Replace final layer of resnet50 with a new layer that has the same number of inputs and 10 outputs for our food categories
resnet50.fc = torch.nn.Linear(num_fc_inputs, num_classes)

#Creates stochastic gradient descent optimizer for final training layer, with a learn rate of 0.001 and a momentum coeffcient of 0.9
#Momentum coeffcient is multiplied by the previous gradient and then added to current gradient to avoid getting trapped in local minima
optimizer = torch.optim.SGD(resnet50.fc.parameters(), lr=0.001, momentum=0.9)

criterion = torch.nn.CrossEntropyLoss()

num_epochs = 10

