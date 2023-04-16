from sklearn.metrics import confusion_matrix
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import torch
import sys
import os


print("Loading validation dataset...")

try:
    with open('pickle/validate.pkl', 'rb') as handle:
        val_dataset = pk.load(handle)
        val_loader = pk.load(handle)
        batch_size = pk.load(handle)
        num_epochs = pk.load(handle)
        classes = pk.load(handle)
except FileNotFoundError:
    print("Test dataset not found. Please run train.py before running evaluation.")
    sys.exit()

print("Loading fine-tuned model...")

try:
    model_dict = torch.load("model/resnet50_food_classification_trained.pth")
except FileNotFoundError:
    print("Fine-tuned model not found. Please run train.py before running evaluation.")
    sys.exit()

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

num_classes = len(classes)

#Number of inputs in the final layer of resnet50
num_fc_inputs = model.fc.in_features

#Replace final layer of resnet50 with a new layer that has the same number of inputs and 10 outputs for our food categories
model.fc = torch.nn.Linear(num_fc_inputs, num_classes)

model.load_state_dict(model_dict)

#Checks if GPU is available for training. If not it default to the CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Creates Cross-Entropy Loss criterion(the loss function)
criterion = torch.nn.CrossEntropyLoss()

val_loss_history = []
val_acc_history = []
y_pred = []
y_true = []

print("Starting evaluation...")
print("\n---------------------------------------------")
print("Number of Epochs: %d" % num_epochs)
print("Batch Size: %d" % batch_size)

for epoch in range(num_epochs):
    #Validation
    model.eval()# Puts the model to evaluation mode
    val_loss = 0
    val_correct = 0
    batch_count = 1
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

            y_pred+=[int(tensor) for tensor in preds]
            y_true+=[int(tensor) for tensor in labels.data]

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct.double() / len(val_loader.dataset)

            val_loss_history.append(val_loss) #Saves a list of the loss for each batch to be plotted following validation
            print('Epoch %d - (Batch %d): Validation Loss: %.4f, Valiation Acc: %.4f' % (epoch+1,batch_count,val_loss, val_acc))
            batch_count+=1
    val_acc_history.append(val_acc)#Saves a list of the accuracy for each epoch to be plotted when training and validation is complete

if os.path.exists("val_results") is not True:
    os.mkdir("val_results")

cm = confusion_matrix(y_true=y_true,y_pred=y_pred)

print("\n\nEvaluation Complete!")
print("----------------------------------------------")
print("Number of Epochs: %d" % num_epochs)
print("Number of Batches per Epoch: %d" % (batch_count-1))
print("Batch Size: %d" % batch_size)

print("\nValidation Loss: %.4f" % val_loss)
print("Validation Accuracy: %.2f%%" % (val_acc*100))

print("\nPlotting Validation data...")

plt.plot(val_loss_history)
plt.title('Validation Loss History')
plt.xlabel('Batches')
plt.ylabel('Validation loss')

plt.savefig("val_results/val_loss.png")

plt.clf()

plt.plot(val_acc_history)
plt.title('Validation Accuracy History')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')

plt.savefig("val_results/val_acc.png")

plt.clf()

fig, ax = plt.subplots(figsize=(num_classes, num_classes))
ax.imshow(cm, cmap=plt.cm.Blues)
ax.set_title('Validation Confusion matrix')
tick_marks = np.arange(num_classes)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, rotation=45)
ax.set_yticklabels(classes)
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, str(cm[i][j]), ha='center', va='center', color='white')
plt.savefig('val_results/confusion_matrix.png')