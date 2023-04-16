import matplotlib.pyplot as plt
import numpy as np


def get_class_accuracy(y_true, y_pred, num_classes):
    class_acc = np.zeros(num_classes)
    class_correct = 0
    for i in range(num_classes):
        for j in range(len(y_true)):
            if y_true[j] == i and y_pred[j] == i:
                class_correct += 1
        class_acc[i] = (class_correct / y_true.count(i))*100
        class_correct = 0
    
    return class_acc

def plot_confusion_matrix(cm, num_classes, classes):
    fig, ax = plt.subplots(figsize=(num_classes, num_classes))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title('Confusion matrix')
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