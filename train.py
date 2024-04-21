import numpy as np
import torch
import torch.nn as nn
import os
import torchvision
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
datt = torchvision.datasets.ImageFolder(root='garbage_classification', transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()]))
datt_loader = torch.utils.data.DataLoader(datt, batch_size=32, shuffle=True)
class_names = datt.classes
#Split the data into 80% train 10 % validation and 10 % test
train_size = int(0.8 * len(datt))
val_size = int(0.1 * len(datt))
test_size = len(datt) - train_size - val_size
train_data, val_data, test_data = torch.utils.data.random_split(datt, [train_size, val_size, test_size])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.base_model = torchvision.models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(2048, len(class_names))
    def forward(self, x):
        return self.base_model(x)
model = model().to(device)
def train(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs} : Loss: {loss.item()}', end='\r')
        model.eval()
        correct, total = 0, 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch + 1}/{epochs} : Accuracy: {100 * correct / total}')
def test(model, test):
    model.eval()
    correct, total = 0, 0
    for images, labels in test:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total}')
train(model, train_loader, val_loader)
test(model, test_loader)