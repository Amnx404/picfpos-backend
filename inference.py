import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import random_split
import random
import os
#import Image
from PIL import Image


device = torch.device("cpu")

classes = [    '3D', 'C0', 'C1', 'C10', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',    'E1', 'E2', 'ENT', 'EX', 'EXB', 'I', 'M', 'MEET', 'MOD', 'R', 'SEW',     'SV', 'TECH', 'WM1', 'WM2']


def load_resnet50_finetuned():
    # Load a pre-trained ResNet50 model
    model_resnet50 = models.resnet50(pretrained=True)

    # Unfreeze some of the layers for fine-tuning
    for name, child in model_resnet50.named_children():
        if name in ['layer3', 'layer4']:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

    # Modify the final layer for  len(dataset.classes) classes
    num_ftrs = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Linear(num_ftrs,  len(classes))

    model_resnet50 = model_resnet50.to(device)

    # Define loss function and optimizer for ResNet50
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_resnet50.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return model_resnet50, optimizer, criterion


def load_resnet101():
    # Load a pre-trained ResNet101 model
    model_resnet101 = models.resnet101(pretrained=True)

    # Modify the final layer for  len(dataset.classes) classes
    num_ftrs = model_resnet101.fc.in_features
    model_resnet101.fc = nn.Linear(num_ftrs,  len(classes))

    model_resnet101 = model_resnet101.to(device)

    # Define loss function and optimizer for ResNet101
    optimizer = optim.Adam(model_resnet101.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return model_resnet101, optimizer, criterion

from PIL import Image
import torchvision.transforms as transforms

chosen_model = load_resnet101()[0]
#load from state dict
chosen_model.load_state_dict(torch.load('./models/FC_Res101_simple/epoch_8.pth', map_location=torch.device('cpu')))


def do_inference(image):
    # Function to preprocess the image
    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #directly take in image file in variable
        # image = Image.open(image)
        
        image = Image.open(image_path)
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        image = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image

    def predict_image(model, image_path, class_names):
        image = preprocess_image(image_path)
        image = image.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted[0].item()]

        return predicted_class

    class_names = classes
    preprocess_image(image)
    p = predict_image(chosen_model, image, class_names)
    # print(p)
    return p


