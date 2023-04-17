from stat_utils import get_class_accuracy, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle as pk
import torch
import sys
import os


learn_rate = 0.001
batch_size = 32
num_epochs = 10


print("\nLoading dataset...")

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

classes = dataset.classes

#Split original dataset into randomly split train(70% of original dataset), Validation(10% of dataset) and test(20% of dataset)
train_size = int(0.7*len(dataset))
val_size = int(0.1*len(dataset))
test_size = int(0.2*len(dataset))
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

# Create data loaders for training, validation, and test sets with a batch size of 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if os.path.exists("pickle") is not True:
    os.mkdir("pickle")

#Saves our validation set, loader, batch size and number of epochs so we can use it later in validate.py
with open('pickle/validate.pkl', 'wb') as handle:
    pk.dump(val_dataset, handle)
    pk.dump(val_loader, handle)
    pk.dump(batch_size, handle)
    pk.dump(num_epochs, handle)
    pk.dump(classes, handle)


#Saves our test set, loader, batch size and number of epochs so we can use it later in test.py
with open('pickle/test.pkl', 'wb') as handle:
    pk.dump(test_dataset, handle)
    pk.dump(test_loader, handle)
    pk.dump(batch_size, handle)
    pk.dump(num_epochs, handle)
    pk.dump(classes, handle)

print("Loading pre-trained model...")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) #Loads pre-trained model ResNet50 with ImageNet weights

#Freezes all pre-trained layers of the original model so they don't get trained
for param in resnet50.parameters():
    param.requires_grad = False

#Our 10 food categories
num_classes = len(dataset.classes)

#Number of inputs in the final layer of resnet50
num_fc_inputs = resnet50.fc.in_features

#Replace final layer of resnet50 with a new layer that has the same number of inputs and 10 outputs for our food categories
resnet50.fc = torch.nn.Linear(num_fc_inputs, num_classes)

#Creates stochastic gradient descent optimizer for final training layer, with a learn rate of 0.001 and a momentum coeffcient of 0.9
#Momentum coeffcient is multiplied by the previous gradient and then added to current gradient to avoid getting trapped in local minima
optimizer = torch.optim.SGD(resnet50.fc.parameters(), lr=learn_rate, momentum=0.9)

#Checks if GPU is available for training. If not it default to the CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Creates Cross-Entropy Loss criterion(the loss function)
criterion = torch.nn.CrossEntropyLoss()


train_loss_history = []
train_acc_epoch_history = []
train_acc_batch_history = []
y_pred = []
y_true = []
print("Starting Training...")
print("\n---------------------------------------------")
print("Number of Epochs: %d" % num_epochs)
print("Batch Size: %d" % batch_size)
print("Learning rate: %.3f" % learn_rate)
print("---------------------------------------------\n")

for epoch in range(num_epochs):
    resnet50.train() # Puts the model to train mode
    train_loss = 0
    train_correct = 0
    batch_count = 0


    #Training
    for inputs, labels in train_loader:
        batch_count+=1
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

        y_pred+=preds
        y_true+=labels.data

        train_loss = train_loss / len(train_loader.dataset) # Calculates mean loss
        train_acc = train_correct.double() / len(train_loader.dataset) #Calculates accuracy

        train_loss_history.append(train_loss) #Saves a list of the loss for each batch to be plotted when training is complete
        train_acc_batch_history.append(train_acc)#Saves a list of the accuracy for each batch

        # Print training results for this batch
        print('Epoch %d - (Batch %d): Training Loss: %.4f, Training Acc: %.4f' % (epoch+1,batch_count,train_loss, train_acc))
        

    train_acc_epoch_history.append(train_acc)#Saves a list of the accuracy for each epoch to be plotted when training is complete

cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
class_acc = get_class_accuracy(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

print("\n\nTraining Complete!")
print("----------------------------------------------")
print("Number of Epochs: %d" % num_epochs)
print("Number of Batches per Epoch: %d" % (batch_count))
print("Batch Size: %d" % batch_size)
print("Learning rate: %.3f" % learn_rate)

print("\nTraining Loss: %.4f" % train_loss)
print("Training Accuracy: %.2f%%" % (train_acc*100))
print("----------------------------------------------")
print("\nClass-wise Accuracy")
print("----------------------------------------------")
for i in range(num_classes):
    print("\"%s\" Accuracy: %.2f%%" % (classes[i], class_acc[i]))
print("----------------------------------------------")

if os.path.exists("train_results") is not True:
    os.mkdir("train_results")

print("\nPlotting Training data...")

plt.plot(train_loss_history)
plt.title('Training Loss History')
plt.xlabel('Batches')
plt.ylabel('Training loss')

plt.savefig("train_results/train_loss.png")

plt.clf()

plt.plot(train_acc_epoch_history)
plt.title('Training Accuracy History')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')

plt.savefig("train_results/train_acc.png")

#Makes and saves a confusion matrix
plot_confusion_matrix(cm, num_classes, classes) #imported from stat_utils.py
plt.savefig('train_results/confusion_matrix.png')

with open('train_results/training_summary.txt', "w") as f:
    f.write("Training Summary\n")
    f.write("----------------------------------------------\n")
    f.write("Number of Epochs: %d\n" % num_epochs)
    f.write("Number of Batches per Epoch: %d\n" % (batch_count))
    f.write("Batch Size: %d\n" % batch_size)
    f.write("\nTotal Training Loss: %.4f\n" % train_loss)
    f.write("Total Training Accuracy: %.2f%%\n" % (train_acc*100))
    f.write("----------------------------------------------\n")
    f.write("\nClass-wise Accuracy\n")
    f.write("----------------------------------------------\n")
    for i in range(num_classes):
        f.write("\"%s\" Accuracy: %.2f%%\n" % (classes[i], class_acc[i]))
    f.write("----------------------------------------------\n")
    f.write("\nTraining History\n")
    f.write("----------------------------------------------\n")
    for epoch in range(num_epochs):
        for batch in range((batch_count)):
            f.write('Epoch %d - (Batch %d): Training Loss: %.4f, Training Acc: %.4f\n' 
                    % (epoch+1,batch+1,train_loss_history[(epoch*batch_count)+batch], train_acc_batch_history[(epoch*batch_count)+batch]))

print("Training data saved to \"train_results\" folder")
print("Saving fine-tuned model...")

if os.path.exists("model") is not True:
    os.mkdir("model")

torch.save(resnet50.state_dict(), "model/resnet50_food_classification_trained.pth") #Saves trained model

print("\nPlease run validate.py to validate the newly trained model.")