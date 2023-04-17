from stat_utils import get_class_accuracy, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle as pk
import torch
import sys
import os


print("\nLoading validation dataset...")

try:
    with open('pickle/validate.pkl', 'rb') as handle:
        val_dataset = pk.load(handle)
        val_loader = pk.load(handle)
        batch_size = pk.load(handle)
        num_epochs = pk.load(handle)
        classes = pk.load(handle)
except FileNotFoundError:
    print("Validation dataset not found. Please run train.py before running evaluation.")
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
print("---------------------------------------------\n")
for epoch in range(num_epochs):
    #Validation
    model.eval()# Puts the model to evaluation mode
    val_loss = 0
    val_correct = 0
    batch_count = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            batch_count+=1
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
            val_acc_history.append(val_acc)#Saves a list of the accuracy for each batch to be plotted when validation is complete
            print('Epoch %d - (Batch %d): Validation Loss: %.4f, Valiation Acc: %.4f' % (epoch+1,batch_count,val_loss, val_acc))
            
    

if os.path.exists("val_results") is not True:
    os.mkdir("val_results")

cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
class_acc = get_class_accuracy(y_true=y_true, y_pred=y_pred,num_classes=num_classes)


print("\n\nEvaluation Complete!")
print("\nValidation Summary")
print("----------------------------------------------")
print("Number of Epochs: %d" % num_epochs)
print("Number of Batches per Epoch: %d" % (batch_count))
print("Batch Size: %d" % batch_size)

print("\nTotal Validation Loss: %.4f" % val_loss)
print("Total Validation Accuracy: %.2f%%" % (val_acc*100))
print("----------------------------------------------")
print("\nClass-wise Accuracy")
print("----------------------------------------------")
for i in range(num_classes):
    print("\"%s\" Accuracy: %.2f%%" % (classes[i], class_acc[i]))
print("----------------------------------------------")

print("\nPlotting Validation data...")

#Makes and saves a confusion matrix
plot_confusion_matrix(cm, num_classes, classes) #imported from stat_utils.py
plt.savefig('val_results/confusion_matrix.png')

with open('val_results/val_summary.txt', "w") as f:
    f.write("Validation Summary\n")
    f.write("----------------------------------------------\n")
    f.write("Number of Epochs: %d\n" % num_epochs)
    f.write("Number of Batches per Epoch: %d\n" % (batch_count))
    f.write("Batch Size: %d\n" % batch_size)
    f.write("\nTotal Validation Loss: %.4f\n" % val_loss)
    f.write("Total Validation Accuracy: %.2f%%\n" % (val_acc*100))
    f.write("----------------------------------------------\n")
    f.write("\nClass-wise Accuracy\n")
    f.write("----------------------------------------------\n")
    for i in range(num_classes):
        f.write("\"%s\" Accuracy: %.2f%%\n" % (classes[i], class_acc[i]))
    f.write("----------------------------------------------\n")
    f.write("\nValidation History\n")
    f.write("----------------------------------------------\n")
    for epoch in range(num_epochs):
        for batch in range((batch_count)):
            f.write('Epoch %d - (Batch %d): Validation Loss: %.4f, Valiation Acc: %.4f\n' 
                    % (epoch+1,batch+1,val_loss_history[(epoch*batch_count)+batch], val_acc_history[(epoch*batch_count)+batch]))

print("\nValidation data saved to \"val_results\" folder")
print("\nPlease run test.py to test model.")