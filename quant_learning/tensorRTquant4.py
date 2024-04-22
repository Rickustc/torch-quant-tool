#自定义op并且完成量化


import torch
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor


class QuantMultiAdd(torch.nn.Module):
    def __int__(self):
        super().init__()
        self._input_quantize = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))   
        self._weight_quantize = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"),axis = 1)   
    
    def forward(self,x,y,z):
        #仅对输入量化
        return self._input_quantize(x) +  self._input_quantize(y)+ self._input_quantize(z)
    
    
    
# quant_modules.initialize()
model = QuantMultiAdd()
model.cuda()



#disable


inputa = torch.randn(1,3,224,224,device="cuda")
inputb = torch.randn(1,3,224,224,device="cuda")
inputc = torch.randn(1,3,224,224,device="cuda")

quant_nn.TensorQuantizer.use_fd_fake_quant = True
torch.onnx.export(model,(inputa,inputb,inputc),"quant_resnet_50.onnx",opset_version=13)
    
    
    
