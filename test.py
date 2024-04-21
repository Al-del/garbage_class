import torch
import os.path
from torchvision import transforms
import torch.nn as nn
import numpy as np
import torchvision
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
path_2 = os.path.join(path, "plastic")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img = torchvision.io.read_image(os.path.join(path_2, "plastic8.jpg"))
img = transform(img)
img = img.unsqueeze(0)

print(img.shape)
print(class_names[predict(model, img)])
