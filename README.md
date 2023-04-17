# Food Classification Deep-Learning Model

## Introduction

This project is a deep learning model for image classification of a set of 10 food items. It uses the ResNet-50 pre-trained on ImageNet database.

## Dataset

The dataset this model was trained on is composed of between 3000 images of a 10 standalone food items scraped from various sources (Pinterest, tumblr, reddit, etc). Each category is composed of 300 images to keep the dataset balanced. Categories of food included in this dataset are:

- Cheeseburger
- Cake
- Cookie
- Fries
- Hotdog
- Pizza
- Salad
- Shrimp
- Steak
- Sushi

The Image dataset is available for download [here.](https://drive.google.com/file/d/1eeGF1GQc97_YIwdqewt6nPmob1bNr7M5) Make sure to save the dataset folder in the same directory as train.py, and test.py 

## Requirements
This project requires the following packages:

- PyTorch 2.0.0
- TorchVision 0.15.1
- Matplotlib 3.5.1
- scikit-learn 1.1.2
- Numpy 1.23.5
- Pytorch GradCAM 1.4.6

## Results
After training the model for 10 epochs with a batch size of 32 and a learning rate of 0.001, we achieved an accuracy of 93.1% on the train set, 91.33% on the validation set, and 93.67% on the test set(as of 04/17/2023). Class-wise accuracies following test are as follows:

- Burger Accuracy: 98.72%
- Cake Accuracy: 85.48%
- Cookie Accuracy: 94.83%
- Fries Accuracy: 92.16%
- Hotdog Accuracy: 95.92%
- Pizza Accuracy: 92.86%
- Salad Accuracy: 95.24%
- Shrimp Accuracy: 95.92%
- Steak Accuracy: 96.55%
- Sushi Accuracy: 88.71%

## License
This project is licensed under the MIT License. See the LICENSE file for details.

 
