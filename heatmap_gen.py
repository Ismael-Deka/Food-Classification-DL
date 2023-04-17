from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import sys
import os


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

dataloader = torch.utils.data.DataLoader(dataset,batch_size=1)

print("Loading fine-tuned model...")

try:
    model_dict = torch.load("model/resnet50_food_classification_trained.pth")
except FileNotFoundError:
    print("Fine-tuned model not found. Please run train.py before running heatmap generator.")
    sys.exit()

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

num_classes = 10

#Number of inputs in the final layer of resnet50
num_fc_inputs = model.fc.in_features

#Replace final layer of resnet50 with a new layer that has the same number of inputs and 10 outputs for our food categories
model.fc = torch.nn.Linear(num_fc_inputs, num_classes)

model.load_state_dict(model_dict)

target_layer = [model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
if os.path.exists("pic_heatmaps") is not True:
    os.mkdir("pic_heatmaps")

class_index = 0
pic_count = 1


for inputs, labels in dataloader:
    if(class_index <= labels[0] and pic_count <= 10):
        if os.path.exists(f"pic_heatmaps/{classes[class_index]}") is not True:
            os.mkdir(f"pic_heatmaps/{classes[class_index]}")
        gs_cam = cam(input_tensor=inputs, targets=[ClassifierOutputTarget(0)])
        image = cv2.imread(f"dataset/{classes[class_index]}/{classes[class_index]}{pic_count}.png" , cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.float32(image) / 255 

        cam_image = show_cam_on_image(image, gs_cam[0], colormap=cv2.COLORMAP_JET)
        print(f"Saving heatmap of {classes[class_index]}...")
        plt.imshow(cam_image)
        plt.axis('off')
        plt.savefig(f"pic_heatmaps/{classes[class_index]}/{classes[class_index]}{pic_count}.png")
        plt.clf()
        pic_count+=1
        
        continue
    if(pic_count > 5):
        pic_count = 1
        class_index+=1
        
    
        

