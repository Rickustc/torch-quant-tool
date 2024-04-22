import torch
import torch.nn as nn
class softmax(nn.Module):
    def __init__(self, in_channel=256,out_channel=512):
        super().__init__()
        self.m = torch.nn.Softmax(dim=3)

    def forward(self, input):
        input = torch.dequantize(input)
        input = self.m(input)
        output = torch.quantize_per_tensor(input,scale = 0.0036, zero_point = 0, dtype=torch.qint8)
        return output
    


x = torch.rand((1,3,224, 224), dtype=torch.float32)
input_data = torch.quantize_per_tensor(x, scale = 0.0036, zero_point = 0, dtype=torch.qint8)  
       
  

export_path= 'softmax.onnx'

model = softmax()


torch.onnx.export(model, (input_data), export_path, opset_version=13,
                      do_constant_folding=True,
                      input_names=['input'], output_names=['output'])