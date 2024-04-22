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
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
    get_tensorrt_backend_config_dict,
    get_qnnpack_backend_config
)
import os
import sys
import numpy as np
from collections import UserDict
from neural_compressor.adaptor.torch_utils.util import input2tuple
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport
# from quantonnxutils import qdq2qlinear
ort = LazyImport('onnxruntime')

def calibrate(model, data_loader,device=None):
  
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            # image = image.to(device)
            if cnt>1:
                break
            print(cnt)
            cnt += 1
            # image = image.to(device)
            model(image)
            break
        
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
    
# modify the backend
q_backend = "qnnpack"  # qnnpack  or fbgemm
torch.backends.quantized.engine = q_backend

# create fp32 model 
model = timm.create_model('resnet50', pretrained=True)
model.eval()

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

# # STEP1: PREPARE
# prepared_model = prepare_fx(model, qconfig_mapping,
#                                 example_inputs=(torch.randn(1, 3, 224, 224),),
#                                 backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config()    
#                                 )
# print('prepare_fx done')

# # STEP2: CALIBRATE
# data_path = '/data1/data/imagenet2012/'
# data_loader,data_loader_test = prepare_data_loaders(data_path=data_path,train_batch_size=30 ,val_batch_size = 100)
# calibrate(prepared_model, data_loader_test)
# print('calibrate done')

# #STEP3: CONVERT
# quantized_fx_model = _convert_fx(prepared_model,
#                             is_reference=False,  # set reference --> False
#                 backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config() 
#                 )
# print('convert_fx done')
import onnx
model = onnx.load("/home/wrq/quant_SDK/torch2onnx_int.onnx")
_input_name_to_nodes = {}
def _get_input_name_to_nodes(nodes):
    """Get input names of nodes."""
    for node in nodes:
        attrs = [attr for attr in node.attribute if attr.type == onnx.AttributeProto.GRAPH \
            or attr.type == onnx.AttributeProto.GRAPHS]
        if len(attrs) > 0:
            for attr in attrs:
                _get_input_name_to_nodes(attr.g.node)
        for input_name in node.input:
            if len(input_name.strip()) != 0:
                if input_name not in _input_name_to_nodes:
                    _input_name_to_nodes[input_name] = [node]
                else:
                    _input_name_to_nodes[input_name].append(node)
_get_input_name_to_nodes(model.graph.node)


def onnx_qlinear_to_qdq(
    model,
    input_name_to_nodes,
):
    """Export ONNX QLinearops model into QDQ model.

    Args:
        model (ModelProto): int8 onnx model.
        input_name_to_nodes (dict): the mapping of tensor name and its destination nodes. 
    """
    from neural_compressor.adaptor.ox_utils.operators import QOPERATORS
    add_nodes = []
    remove_nodes = []
    inits = []
# if check_model(model):
    for node in model.graph.node:
        import pdb
        pdb.set_trace()
        if node.op_type in QOPERATORS:
            if node.output[0] not in input_name_to_nodes:
                continue
            children = []
            for out in node.output:
                children.extend(input_name_to_nodes[node.output[0]])
            converter = QOPERATORS[node.op_type](
                node,
                children,
                model.graph.initializer)
            done, add_node, init = converter.convert()
            if done:
                add_nodes.extend(add_node)
                inits.extend(init)
                remove_nodes.append(node)
                print(add_nodes)
    # return add_nodes, remove_nodes, inits
onnx_qlinear_to_qdq(model,_input_name_to_nodes)

def is_int8_model(model):
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

# def _prepare_inputs(pt_model, input_names, example_inputs):
#     """Prepare input_names and example_inputs."""
#     if isinstance(example_inputs, dict) or isinstance(example_inputs, UserDict):
#         input_names = input_names or list(example_inputs.keys())
#         if isinstance(example_inputs, UserDict):
#             example_inputs = dict(example_inputs)
#     # match input_names with inspected input_order, especailly for bert in hugginface.
#     elif input_names and len(input_names) > 1:
#         import inspect
#         input_order = inspect.signature(pt_model.forward).parameters.keys()
#         flag = [name in input_order for name in input_names] # whether should be checked
#         if all(flag):
#             new_input_names = []
#             new_example_inputs = []
#             for name in input_order:
#                 if name in input_names:
#                     new_input_names.append(name)
#                     id = input_names.index(name)
#                     new_example_inputs.append(example_inputs[id])
#             input_names = new_input_names
#             example_inputs = new_example_inputs
#         example_inputs = input2tuple(example_inputs)
#     return input_names, example_inputs

# # print(is_int8_model(quantized_fx_model))
# assert is_int8_model(quantized_fx_model), "The exported model is not INT8 model, "\
#     "please reset 'dtype' to 'FP32' or check your model."
    
    
# input_names=["images"]
# output_names=["pred"]
# opset_version=14
# example_inputs = torch.rand([100,3,224,224])
# input_names, example_inputs = _prepare_inputs(quantized_fx_model, input_names, example_inputs)


# def model_wrapper(model_fn):
#     # export doesn't support a dictionary output, so manually turn it into a tuple
#     # refer to https://discuss.tvm.apache.org/t/how-to-deal-with-prim-dictconstruct/11978
#     def wrapper(*args, **kwargs):
#         output = model_fn(*args, **kwargs)
#         if isinstance(output, dict):
#             return tuple(v for v in output.values() if v is not None)
#         else:
#             return output
#     return wrapper
# quantized_fx_model.forward = model_wrapper(quantized_fx_model.forward)

# save_path = "torch2onnx_int.onnx"
# with torch.no_grad():
#     try:
#         torch.onnx.export(
#             quantized_fx_model,
#             input2tuple(example_inputs),
#             save_path, 
#             opset_version=opset_version,
#             input_names=input_names,
#             output_names=output_names,
#             )
#     except Exception as e:

#         logger.error("Export failed, possibly because unsupported quantized ops. Check " 
#                         "neural-compressor/docs/source/export.md#supported-quantized-ops "
#                         "for supported ops.")
#         logger.error("Please fallback unsupported quantized ops by setting 'op_type_dict' or "
#                         "'op_name_dict' in  config. "
#                         )
     

# sess_options = ort.SessionOptions()
# sess_options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# sess_options.optimized_model_filepath=save_path
# ort.InferenceSession(save_path, sess_options)
# print("torch2onnxdone")
# #STEP4: EXPORT
# exported_model_path = './export/resnet50_quant_int8_linear.onnx'
# image_tensor = torch.rand([1,3,224,224])
# torch.onnx.export(quantized_fx_model , 
#                   image_tensor, 
#                   exported_model_path, 
#                   verbose=True, 
#                   input_names=["images"], 
#                   output_names=["pred"], 
#                   opset_version=16)
# print('export done')

# #STEP5 QDQ to Qlinear model
# qdqmodel_path = './export/qlinear_resnet50_quant_int8_linear.onnx'
# # qdq2qlinear(exported_model_path,qdqmodel_path)
# print('qlinear done')







   

    
    


