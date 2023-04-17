from stat_utils import get_class_accuracy, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle as pk
import torch
import sys
import os

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

#Number of inputs in the final layer of resnet50
num_fc_inputs = model.fc.in_features

#Replace final layer of resnet50 with a new layer that has the same number of inputs and 10 outputs for our food categories
model.fc = torch.nn.Linear(num_fc_inputs, num_classes)

model.load_state_dict(model_dict)

#Checks if GPU is available for training. If not it default to the CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Creates Cross-Entropy Loss criterion(the loss function)
criterion = torch.nn.CrossEntropyLoss()

test_loss_history = []
test_acc_history = []
y_pred = []
y_true = []

print("Starting Testing...")
print("\n---------------------------------------------")
print("Batch Size: %d" % batch_size)
print("---------------------------------------------\n")

#Test
model.eval()
test_loss = 0
test_correct = 0
batch_count = 0
with torch.no_grad():
        for inputs, labels in test_loader:
            batch_count+=1
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data)

            y_pred+=[int(tensor) for tensor in preds]
            y_true+=[int(tensor) for tensor in labels.data]

            test_loss = test_loss / len(test_loader.dataset)
            test_acc = test_correct.double() / len(test_loader.dataset)

            test_loss_history.append(test_loss) #Saves a list of the loss for each batch to be plotted following test
            test_acc_history.append(test_acc)#Saves a list of the accuracy for each batch to be plotted when test is complete
            print('Batch %d: Test Loss: %.4f, Test Acc: %.4f' % (batch_count,test_loss, test_acc))
            
    

if os.path.exists("test_results") is not True:
    os.mkdir("test_results")

cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
class_acc = get_class_accuracy(y_true=y_true, y_pred=y_pred,num_classes=num_classes)


print("\n\nTesting Complete!")
print("\nTest Summary")
print("----------------------------------------------")
print("Batch Size: %d" % batch_size)

print("\nTotal Test Loss: %.4f" % test_loss)
print("Total Test Accuracy: %.2f%%" % (test_acc*100))
print("----------------------------------------------")
print("\nClass-wise Accuracy")
print("----------------------------------------------")
for i in range(num_classes):
    print("\"%s\" Accuracy: %.2f%%" % (classes[i], class_acc[i]))
print("----------------------------------------------")

print("\nPlotting Test data...")

#Makes and saves a confusion matrix
plot_confusion_matrix(cm, num_classes, classes) #imported from stat_utils.py
plt.savefig('test_results/confusion_matrix.png')

with open('test_results/test_summary.txt', "w") as f:
    f.write("Test Summary\n")
    f.write("----------------------------------------------\n")
    f.write("Batch Size: %d\n" % batch_size)
    f.write("\nTotal Test Loss: %.4f\n" % test_loss)
    f.write("Total Test Accuracy: %.2f%%\n" % (test_acc*100))
    f.write("----------------------------------------------\n")
    f.write("\nClass-wise Accuracy\n")
    f.write("----------------------------------------------\n")
    for i in range(num_classes):
        f.write("\"%s\" Accuracy: %.2f%%\n" % (classes[i], class_acc[i]))
    f.write("----------------------------------------------\n")
    f.write("\nTest History\n")
    f.write("----------------------------------------------\n")

    for batch in range((batch_count)):
            f.write('Batch %d: Test Loss: %.4f, Test Acc: %.4f\n' 
                    % (batch+1,test_loss_history[batch], test_acc_history[batch]))

print("\nTest data saved to \"test_results\" folder")