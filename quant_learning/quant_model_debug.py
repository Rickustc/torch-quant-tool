import torch
import torch.fx
from torch.fx.node import Node
from typing import Dict
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
 
 
 
def torch_cosine_similarity(result1,result2,dim=0,eps=1e-6):
    cos = torch.nn.CosineSimilarity(dim=dim,eps=eps)
    output = cos(result1,result2)
    return output
 
 
def calibrate(model, image_tensor):
    with torch.no_grad():
            model(image_tensor)

 
#属于类似编译器中的passes，进行图的修改等操作
class CosineSimilarityProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
 
    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}
        env_fp32 : Dict[str, Node] = {}
        debug_value = {}
        check_msg = {} 

        def load_arg(a):
            "map_arg: Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys."
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])
        
        def load_arg_fp32(a):
            return torch.fx.graph.map_arg(a, lambda n: env_fp32[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr
        

        print("quant similarity : ")
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                # input
                result = next(args_iter)
                result_fp32 = result
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
                # if "quantize" in node.name:
                #     result_fp32 = load_arg_fp32(node.args)[0]
                #     debug_value[node.name] = load_arg(node.args)[0]
                # else:
                #     # 对于add和cat来说也需要执行相应的部分
                #     result_fp32 = node.target(*load_arg_fp32(node.args), **load_arg_fp32(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                op_sim = self.modules[node.target]
    
                result_fp32_layer = op_sim.forward_fp32(*load_arg(node.args), **load_arg(node.kwargs))
                result_int8_layer = op_sim(*load_arg(node.args), **load_arg(node.kwargs))
                result_fp32_model = op_sim.forward_fp32(*load_arg(node.args), **load_arg(node.kwargs))
                activation_dif_accmulated = torch_cosine_similarity(result_int8_layer, result_fp32_model)
                activation_dif_layer = torch_cosine_similarity(result_int8_layer, result_fp32_layer)
                weight_dif = torch_cosine_similarity(op_sim.get_weight(), op_sim.get_weight())
                # result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
                
            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype
            env[node.name] = result

        return load_arg(result)
    
    
class QuantAccuracyCheck:
    """
    QuantAccuracy propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        
    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}
        env_fp32 : Dict[str, Node] = {}
        debug_value = {}
        check_msg = {} 
    
        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])
        
        def load_arg_fp32(a):
            return torch.fx.graph.map_arg(a, lambda n: env_fp32[n.name])
        
        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr
        
        print("quant similarity : ")
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                # input
                result = next(args_iter)
                result_fp32 = result
            elif node.op == 'get_attr': 
                result = fetch_attr(node.target)
            elif node.op == 'call_function':   # quantize or add/cat
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
                if "quantize" in node.name:
                    import pdb
                    pdb.set_trace()
                    result_fp32 = load_arg_fp32(node.args)[0]
                    debug_value[node.name] = load_arg(node.args)[0]
                else:
                    # 对于add和cat来说也需要执行相应的部分
                    result_fp32 = node.target(*load_arg_fp32(node.args), **load_arg_fp32(node.kwargs))
            elif node.op == 'call_method':   # dequantize
                self_obj, *args = load_arg(node.args)
        
   
    

    
   
import timm 
model = timm.create_model('resnet50', pretrained=True)
model.eval()


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
                                example_inputs=(torch.randn(1, 3, 256, 256),),
                                backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config()    
                                )

image_tensor = torch.rand([1,3,224,224])
calibrate(prepared, image_tensor)


quantized_fx = _convert_fx(prepared, 
            is_reference=True,  # 选择reference模式为False
            backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config() 
            )


gm = torch.fx.symbolic_trace(quantized_fx)

CosineSimilarityProp(gm).propagate(image_tensor)

    

    
