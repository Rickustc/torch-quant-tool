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
from pdb import set_trace as bp
import onnx
from onnx_qdq_to_qlinear import ort_infer_check,onnx_qdq_to_qlinear
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
    get_tensorrt_backend_config_dict,
    get_qnnpack_backend_config
)
from utils import prepare_data_loaders, calibrate, print_size_of_model, evaluate

import logging
logger = logging.getLogger("neural_compressor")

def is_int8_weight(model):
    """Check whether the input model is a int8 model.

    Args:
        model (torch.nn.Module): input model

    Returns:
        result(bool): Return True if the input model is a int8 model.
    """
    def _is_int8_value(value):
        """Check whether the input tensor is a int8 tensor."""
        if hasattr(value, 'dtype') and 'int8' in str(value.dtype):
            return True
        else:
            return False

    stat_dict = dict(model.state_dict())
    for name, value in stat_dict.items():
        if _is_int8_value(value):
            return True
        # value maybe a tuple, such as 'linear._packed_params._packed_params'
        if isinstance(value, tuple):
            for v in value:
                if _is_int8_value(v):
                    return True
    return False


def is_int8_activation(model):
    #check activation int8

    input_data = torch.randn(1, 3, 256, 256)
    # hook for getting the output shape ofeach layer
    def print_layer_output(module, input, output):
        # print(f"Output of {module.__class__.__name__}: {output.dtype}")
        assert output.dtype==torch.qint8, "activation should be int8"
        
    hooks = []
    for layer in model.named_children():
    
        hook = layer[1].register_forward_hook(print_layer_output)
        hooks.append(hook)
        if isinstance(layer, nn.Module):
            hook = layer.register_forward_hook(print_layer_output)
            hooks.append(hook)
    output = model(input_data)
    for hook in hooks:
        hook.remove()
    return True


# 
def create_model(name,is_pt=True):
    model = timm.create_model(name, pretrained=is_pt)
    return model


def quant(
    model,quant_mode,example_inputs,calib_dataloader,calib_count
    ):
    # modify the backend
    model.eval()
    model.cuda()
    q_backend = "qnnpack"  
    torch.backends.quantized.engine = q_backend
    
    
    # set quant config
    qconfig = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=torch.ao.quantization.observer.default_per_channel_weight_observer
    )
    # qnnpakc not support perchannel for linear oop now
    qconfig_linear = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)   
    )
    qconfig_mapping = (QConfigMapping()
        .set_global(qconfig)  
        .set_object_type(torch.nn.Linear, qconfig_linear)  
    )
    
    
    # STEP1: PREPARE
    prepared_model = prepare_fx(model, qconfig_mapping,
                                example_inputs=example_inputs,
                                backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config()    
                                )
    print('prepare_fx done')
    
    # STEP2: CALIBRATE
    # -1 means go through the whole dataset
    
    logger.info(f"Start calibrate, model is in {next(prepared_model.parameters()).device}")
    calibrate(prepared_model, calib_dataloader, torch.device('cuda'), count = calib_count)
    prepared_model.cpu()
    print('calibrate done')

    #STEP3: CONVERT
    quantized_fx_model = _convert_fx(prepared_model,
                                is_reference=False,  # set reference --> False
                    backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config() 
                    )
    print('convert_fx done')
    
    
    # check weight int8
    assert is_int8_weight(quantized_fx_model)==True, "The exported model is not INT8 weight, "\
    "please reset 'dtype' to 'FP32' or check your model."
    # check activation int8
    assert is_int8_activation(quantized_fx_model)==True, "The exported model is not INT8 activation, "\
    "please reset 'dtype' to 'FP32' or check your model."
    
    
    
    
    def show_qat_eval(qat_model):
        pass
    
    def show_ptq_eval(ptq_model):
        pass

    def show_qat_eval(qat_model):
        pass
    
    return quantized_fx_model


def torch_to_int8_onnx(
    int8_model,
    example_input,
    save_path,
    input_names=["images"],
    output_names=["pred"],
    opset_version=14
    ):
    os.makedirs("evas_workspace",exist_ok=True)
    tmp_path = "./evas_workspace"+"/tmp_model.onnx"
    with torch.no_grad():
        try:
            torch.onnx.export(
                int8_model,
                example_input,
                tmp_path, 
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                verbose=False,
                )
        except:
            raise
    
    return onnx_qdq_to_qlinear(tmp_path,save_path)
        
    

        
    
    

    
