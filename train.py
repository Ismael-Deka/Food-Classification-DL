import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as T
import pickle as pk

from sklearn.model_selection import train_test_split

food_categories = ['burger', 'cake', 'cookie', 'fries', 'hotdog', 'pizza', 'salad', 'shrimp', 'steak', 'sushi']

labels = []

for x in food_categories:
    labels.extend([x for _ in range(300)])


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize images
])

dataset = ImageFolder(root='dataset', transform=data_transforms)


X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

with open('test.pickle', 'wb') as handle:
    pk.dump(X_test, handle)
    pk.dump(y_test, handle)


resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
