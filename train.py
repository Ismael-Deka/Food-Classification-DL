import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch
import pickle as pk
import sys
import matplotlib.pyplot as plt

food_categories = ['burger', 'cake', 'cookie', 'fries', 'hotdog', 'pizza', 'salad', 'shrimp', 'steak', 'sushi']

print("Loading dataset...")

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize images
])

try:
    dataset = ImageFolder(root='dataset', transform=data_transforms)
except FileNotFoundError:
    print("Dataset not found.")
    sys.exit()

#Split original dataset into randomly split train(70% of original dataset), Validation(10% of dataset) and test(20% of dataset)
train_size = int(0.7*len(dataset))
val_size = int(0.1*len(dataset))
test_size = int(0.2*len(dataset))
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

# Create data loaders for training, validation, and test sets with a batch size of 32
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


#Saves our test set and loader so we can use it later in test.py
with open('test.pickle', 'wb') as handle:
    pk.dump(test_dataset, handle)
    pk.dump(test_loader, handle)

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

#Checks if GPU is available for training. If not it default to the CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Creates Cross-Entropy Loss criterion(the loss function)
criterion = torch.nn.CrossEntropyLoss()

#How many times the model will train on the dataset
num_epochs = 10

train_loss_history = []
train_acc_history = []

print("Training model...")

for epoch in range(num_epochs):
    resnet50.train() # Puts the model to train mode
    train_loss = 0
    train_correct = 0
    batch_count = 1
    for inputs, labels in train_loader:
        inputs = inputs.to(device) #loads input on to GPU if available
        labels = labels.to(device) #loads labels on to GPU if available
        
        optimizer.zero_grad() #Sets gradients to zero at the start of each batch
        outputs = resnet50(inputs) #Feeds batch input into the model returns output probilities(10 for each food item)
        loss = criterion(outputs, labels) #Finds the loss between the model's output and the true label.
        loss.backward() #Back-propagation. Computes the gradients of the loss
        optimizer.step() #Updates model using the gradients


        train_loss += loss.item() * inputs.size(0)  #Calculates a running total for loss for each epoch. 
                                                    #Loss for a single input * batch size + total loss so far


        _, preds=torch.max(outputs, 1) #Find output with the highest probablity and save it as the predicted value for the food item.
        train_correct += torch.sum(preds == labels.data) # Counts how many predictions matched the true values

        train_loss = train_loss / len(train_loader.dataset) # Calculates mean loss
        train_acc = train_correct.double() / len(train_loader.dataset) #Calculates accuracy

        train_loss_history.append(train_loss) #Saves a list of the loss for each batch to be plotted when training is complete
        

        # Print results for this epoch/batch
        print('Epoch %d - (Batch %d): Training Loss: %.4f, Training Acc: %.4f' % (epoch+1,batch_count,train_loss, train_acc))
        batch_count+=1
    train_acc_history.append(train_acc)#Saves a list of the accuracy for each epoch to be plotted when training is complete

print("\n\nTraining Complete!")
print("----------------------------------------------")
print("Number of Epochs: %d" % num_epochs)
print("Number of Batches per Epoch: %d" % batch_count-1)
print("Batch Size: %d" % batch_size)
print("Learning rate: %f" % 0.001)

print("\nTraining Loss: %.4f" % train_loss)
print("Training Accuracy: %.2f%%" % (train_acc*100))

print("\nPlotting Training data...")

plt.plot(train_loss_history)
plt.xlabel('Batches')
plt.ylabel('Training loss')

plt.savefig("train_loss.png")

plt.clf()

plt.plot(train_acc_history)
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')

plt.savefig("train_acc.png")
print("Saving trained model...")

torch.save(resnet50.state_dict(), "resnet50_food_classification_trained.pth") #Saves trained model