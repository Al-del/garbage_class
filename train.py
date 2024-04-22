import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from efficientnet_v2 import EfficientNetV2

import os
import torchvision
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images,


])
datt = torchvision.datasets.ImageFolder(root='garbage_classification', transform=transform)
class_count = [i for i in torch.bincount(torch.tensor(datt.targets))]
class_weights = 1./torch.tensor(class_count, dtype=torch.float)
sample_weights = class_weights[datt.targets]
sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Create the data loader with shuffling
datt_loader = torch.utils.data.DataLoader(datt, batch_size=batch_size, sampler=sampler)#Print the first batch of the data loader
images, labels = next(iter(datt_loader))
print(images.shape)
print(labels)
class_names = datt.classes
#Split the data into 80% train 10 % validation and 10 % test
train_size = int(0.8 * len(datt))
val_size = int(0.1 * len(datt))
test_size = len(datt) - train_size - val_size
train_data, val_data, test_data = torch.utils.data.random_split(datt, [train_size, val_size, test_size])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.base_model = EfficientNetV2(model_name='b1',
                                         in_channels=3,
                        n_classes=12,
                        pretrained=True)
        #Flatten the output of the base model
        self.flatten = nn.Flatten()
        #Add a dropout layer
        self.dropout = nn.Dropout(0.5)
        #Add a dense layer with activation function SOftmax
        self.fc = nn.Linear(1000, 12)  # Change the input size to 1000
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
#model = model().to(device)
def train(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize variables for early stopping
    patience = 3
    best_loss = None
    best_epoch = None
    counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(output, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            pbar.set_postfix({'loss': running_loss / total, 'accuracy': running_corrects.double().item() / total})
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs} : Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Early stopping check
        if best_loss is None:
            best_loss = epoch_loss
            best_epoch = epoch
        elif epoch_loss > best_loss:
            counter += 1
            print(f'EarlyStopping counter: {counter} out of {patience}')
            if counter >= patience:
                print(f'Early stopping at epoch: {best_epoch}')
                return
        else:
            best_loss = epoch_loss
            best_epoch = epoch
            counter = 0

        test(model, val_loader)
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
class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        self.base_model = EfficientNetV2(model_name='b1',
                                         in_channels=3,
                        n_classes=12,
                        pretrained=True)
        #EfficientNetV2(model_name='b1',
                                         #in_channels=3,
                     #   n_classes=12,
                     #   pretrained=True)
        #Add a flatten layer
        self.flatten = nn.Flatten()
        #Add a dropout layer
        self.dropout = nn.Dropout(0.5)
        #Add a dense layer with activation function SOftmax
        self.fc = nn.Linear(12, 12)
        self.activation = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
model = model_2().to(device)
train(model, train_loader, val_loader, epochs=10)
test(model, test_loader)
torch.save(model.state_dict(), 'model_garbage.pth')

def plot_confusion_matrix(model, test):
        model.eval()
        y_pred = []
        y_true = []
        for images, labels in test:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(include_values=True, cmap='viridis', ax=plt.gca())
        plt.show()
plot_confusion_matrix(model, test_loader)