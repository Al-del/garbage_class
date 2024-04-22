import os.path
from torchvision import transforms

import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
path = "/home/informatica-pentru-viitor/Desktop/garbage_class_pred/garbage_classification"
class_names= ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        self.base_model = torchvision.models.efficientnet_b1(pretrained=True)
        self.base_model._fc = nn.Linear(1280, 12)
    def forward(self, x):
        return self.base_model(x)
model = model_2()
model.load_state_dict(torch.load("./model_garbage.pth", map_location=torch.device('cpu')))

def predict(model, input_tensor):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Temporarily set all the requires_grad flag to false
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)  # Get the index of the max log-probability
    return predicted.item()

cap = cv2.VideoCapture(-1)  # 0 for default camera, or replace with video file path
while True:
    # Read frame by frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to size 224, 224, 3
    resized_frame = cv2.resize(frame, (224, 224))

    # Add an extra dimension to represent the batch size
    resized_frame = np.expand_dims(resized_frame, axis=0)

    # Fit the resized frame into the model
    prediction = predict(model, torch.Tensor(resized_frame).permute(0, 3, 1, 2))
    # Print the class predicted
    print(class_names[prediction])
    cv2.imshow('Frame', frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()