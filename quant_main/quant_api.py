import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import timm
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx,_convert_fx,convert_to_reference_fx,prepare_qat_fx
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
import onnxruntime as ort
from utils import prepare_data_loaders, calibrate, print_size_of_model, evaluate,_get_module_op_stats,TopK,Dataloader

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
        """Check whether the input tensor is a qint8 tensor."""
        if hasattr(value, 'dtype') and 'qint8'  in str(value.dtype):  #
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
    
    # assert quant_mode in ["ptq,qat"]
    
    if quant_mode=="ptq":
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
        print('prepare_ptq_fx done')
        
        # STEP2: CALIBRATE
        logger.info(f"Start calibrate, model is in {next(prepared_model.parameters()).device}")
        # -1 means go through the whole dataset
        calibrate(prepared_model, calib_dataloader, torch.device('cuda'), count = calib_count)
        prepared_model.cpu()
        print('calibrate done')

        #STEP3: CONVERT
        quantized_fx_model = _convert_fx(prepared_model,
                                    is_reference=False,  # set reference --> False
                        backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config() 
                        )
        print('convert_ptq_fx done')
        
        
        # check wether weight is int8
        assert is_int8_weight(quantized_fx_model)==True, "The exported model is not INT8 weight, "\
        "please reset 'dtype' to 'FP32' or check your model."
        # check wether activation is int8
        assert is_int8_activation(quantized_fx_model)==True, "The exported model is not INT8 activation, "\
        "please reset 'dtype' to 'FP32' or check your model."
    else:
        pass


    # _get_module_op_stats(quantized_fx_model)
    # _dump_model_op_stats(quantized_fx_model)
    
    
    
    # def show_qat_eval(qat_model):
    #     pass
    
    # def show_ptq_eval(ptq_model):
    #     pass

    # def show_qat_eval(qat_model):
    #     pass
    
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


def eval_onnxmodel(onnxmodel):
    top1 = TopK()
    top1.reset()
    #set a specify dataloader
    dataloader = Dataloader("/data1/data/imagenet2012/dataILSVRC2012_img_val", "/data1/data/imagenet2012/val.txt", batch_size=100)
    sess = ort.InferenceSession(onnxmodel.SerializeToString(), providers=ort.get_available_providers())
    input_names = [i.name for i in sess.get_inputs()]
    print("eval onnx int8 model")
    for input_data, label in tqdm(dataloader):
        output = sess.run(None, dict(zip(input_names, [input_data])))
        top1.update(output, label)
    return top1.result()

def cos_similarity_bettween_model(fx_model,onnx_model_path,input_tensor):
    # get torchmodel output
    import numpy as np
    # use dataloader tensor as input tensor
    fx_output = fx_model(input_tensor)
    fx_output = fx_output.numpy()
    # get onnxmodel output
    onnx_model = ort.InferenceSession(onnx_model_path)
    onnx_input = {onnx_model.get_inputs()[0].name: input_tensor.numpy()}
    onnx_output = onnx_model.run(None, onnx_input)[0]
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(fx_output)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(onnx_output)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(fx_output-onnx_output)
 
   
    dot_product = np.dot(fx_output, onnx_output.T)
    
    norm_b = np.linalg.norm(fx_output)
    norm_a = np.linalg.norm(onnx_output.T)
    
    
    cosine_similarity = dot_product / (norm_a * norm_b)
    return cosine_similarity


onnx_qdq_to_qlinear("/home/wrq/vit_quant.onnx","vit_int8_.onnx")
        
        
# import onnx
# import numpy as np
# import onnx.helper as helper
# model = onnx.load("/home/wrq/yolov8_relu_ptr_Q.onnx")
# # for node in model.graph.initializer:
# node_name = ['/Concat_quant', '/Concat_1_quant', '/Concat_2_quant', '/Concat_3_quant', '/Concat_4_quant', '/Concat_5_quant', '/Concat_6_quant', '/Concat_7_quant', '/Concat_8_quant', '/Concat_9_quant', '/22/Concat_quant', '/Concat_10_quant', 
#  '/Concat_11_quant', '/22/Concat_1_quant', '/Concat_12_quant', '/22/Concat_4_quant', '/22/Concat_5_quant', '/22/Concat_2_quant']

# for item in model.graph.initializer:
#     print(item.name)
#     if item.name in node_name:
#         import pdb
#         pdb.set_trace()
#         print(item)
        
# from onnx.shape_inference import infer_shapes
# from onnx.shape_inference import infer_shapes
# from onnx import load_model, save_model
# import torch
# import torch.nn as nn
# import numpy as np

# onnx_model = load_model("/home/wrq/yolov8_int8_.onnx")
# # onnx_model = infer_shapes(onnx_model)
# # save_model(onnx_model, "infered_shape.onnx")
# for node in onnx_model.graph.node:
#     if node.op_type == "Concat":
#         print(node.attribute)
        



    # print("shape: ", node.dims)
    # weight = np.frombuffer(node.raw_data, dtype=np.float32).reshape(*node.dims)
    # print(weight)

        
    
    

    
