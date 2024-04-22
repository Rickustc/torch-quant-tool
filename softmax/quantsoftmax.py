import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import timm
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx,_convert_fx,convert_to_reference_fx
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
from SelfAttention import *
import onnxruntime as ort
import copy
import os
from neural_compressor.adaptor.ox_utils.quantizer import Quantizer
import onnx
from neural_compressor.adaptor.ox_utils.operators import OPERATORS
from neural_compressor.model.onnx_model import ONNXModel


def calibrate(model, random_tensor):
    model(random_tensor)
        
def prepare_data_loaders(data_path,train_batch_size,val_batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler,num_workers=32)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=val_batch_size,
        sampler=test_sampler,num_workers=32)

    return data_loader, data_loader_test
    
q_backend = "qnnpack"  # qnnpack  or fbgemm
torch.backends.quantized.engine = q_backend

hidden_dim = 64
model = SelfAttention(hidden_dim)

model.eval()
random_tensor = torch.randn(1, 10, hidden_dim)  # 输入形状为(batch_size, sequence_length, hidden_dim)

qconfig = torch.ao.quantization.qconfig.QConfig(
    activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
    weight=torch.ao.quantization.observer.default_per_channel_weight_observer
    
    
)
qconfig_linear = torch.ao.quantization.qconfig.QConfig(
    activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
    weight=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)   # qnnpakc not support perchannel for linear oop now
)
qconfig_mapping = (QConfigMapping()
    .set_global(qconfig)  
    .set_object_type(torch.nn.Linear, qconfig_linear)  
)


prepared = prepare_fx(model, qconfig_mapping,
                                example_inputs=(random_tensor),
                                backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config()    
                                )




calibrate(prepared, random_tensor)

print("###################################")

quantized_fx = _convert_fx(prepared, 
            is_reference=False,  # 选择reference模式为False
            backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config() 
            )

exported_model_path = 'quant_softmax_int8.onnx'
image_tensor = random_tensor

torch.onnx.export(quantized_fx, image_tensor, exported_model_path,verbose=False,operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,input_names=["images"], output_names=["pred"], opset_version=13)
