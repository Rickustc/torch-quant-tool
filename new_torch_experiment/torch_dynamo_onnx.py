import torch
import torch.nn as nn
class MLPModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc0 = nn.Linear(8, 8, bias=True)
      self.fc1 = nn.Linear(8, 4, bias=True)
      self.fc2 = nn.Linear(4, 2, bias=True)
      self.fc3 = nn.Linear(2, 2, bias=True)

  def forward(self, tensor_x: torch.Tensor):
      tensor_x = self.fc0(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      tensor_x = self.fc1(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      tensor_x = self.fc2(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      output = self.fc3(tensor_x)
      return output

model = MLPModel()
tensor_x = torch.rand((97, 8), dtype=torch.float32)
export_output = torch.onnx.dynamo_export(model, tensor_x)

export_output.save("mlp.onnx")