import tf2onnx
import tensorflow as tf
import torch
import onnx
from onnx2pytorch import ConvertModel

# Convert the model to ONNX format
loadd_model = tf.keras.models.load_model('my_keras_model.h5')
onnx_model, _ = tf2onnx.convert.from_keras(loadd_model)
pytorch_model = ConvertModel(onnx_model)
#Generate a class who has the same structure as the model
class model_2(torch.nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        self.base_model = pytorch_model
    def forward(self, x):
        return self.base_model(x)
#Save the class and the model
model = model_2()
torch.save(model, 'model_2.pth')