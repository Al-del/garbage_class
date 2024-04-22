import os.path
from torchvision import transforms

import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

from efficientnet_v2 import EfficientNetV2

path = "/home/informatica-pentru-viitor/Desktop/garbage_class_pred/garbage_classification"
class_names= ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images,


])
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
model = model_2()
model.load_state_dict(torch.load('./model_garbage.pth', map_location=torch.device('cpu')))
def predict(model, input_tensor):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Temporarily set all the requires_grad flag to false
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)  # Get the index of the max log-probability
    return predicted.item()

cap = cv2.VideoCapture(0)  # 0 for default camera, or replace with video file path
while True:
    # Read frame by frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to size 224, 224, 3
    #Use the transform to convert the image to tensor
    frame_pil = Image.fromarray(frame)

    # Apply the transform to the frame
    input_tensor = transform(frame_pil)

    # Add an extra dimension for batch size
    input_tensor = input_tensor.unsqueeze(0)

    # Make a prediction
    prediction = predict(model, input_tensor)
    print(class_names[prediction])
    cv2.imshow('Frame', frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()