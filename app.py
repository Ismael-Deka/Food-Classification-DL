import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize images
])

data_dir = 'food_images'
dataset = ImageFolder(root=data_dir, transform=data_transforms)

resnet50 = models.resnet50(pretrained=True)