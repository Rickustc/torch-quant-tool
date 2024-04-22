import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx,_convert_fx
from torch.ao.quantization import QConfigMapping
import torch.ao as ao
import torchvision.transforms as transforms
import torch.nn as nn
from torch.ao.quantization.observer import *
import os
# Set up warnings
import warnings
from tqdm import tqdm, trange
import time
import pdb
from pdb import set_trace as bp
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
    # get_evas_backend_config,
    get_tensorrt_backend_config_dict,
    get_qnnpack_backend_config
)
from torch.onnx import OperatorExportTypes    
import copy
from torch.fx.passes.graph_drawer import FxGraphDrawer
import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler




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

qconfig = torch.ao.quantization.qconfig.QConfig(
    activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
    weight=torch.ao.quantization.observer.default_per_channel_weight_observer
)
qconfig_mapping = (QConfigMapping()
    .set_global(qconfig)  
)


prepared = prepare_fx(model, qconfig_mapping,
                                example_inputs=(torch.randn(1, 3, 224, 224),),
                                backend_config= get_tensorrt_backend_config_dict() 
)

prepared(tensor_x)


quantized_fx = _convert_fx(prepared, 
            is_reference = True,  # 选择reference模式
            qconfig_mapping = qconfig_mapping,
            backend_config =  get_tensorrt_backend_config_dict()    #get_qnnpack_backend_config()  
            )
onnx_program = torch.onnx.dynamo_export(quantized_fx , tensor_x)