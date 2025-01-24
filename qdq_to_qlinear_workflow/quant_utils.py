from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
from PIL import Image
import numpy as np
from scipy import spatial
import os
from typing import Callable, Dict, List, Optional, Sequence, Union
import logging
import math
from onnxruntime.quantization.onnx_model import ONNXModel
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.quant_utils import (
    DEQUANT_OP_NAME,
    DEQUANT_OUTPUT_SUFFIX,
    QUANT_INPUT_SUFFIX,
    TENSOR_NAME_QUANT_SUFFIX,
    find_by_name,
    load_model_with_shape_infer,
)
from pathlib import Path

def get_cosine_dist(x, y):
    cosine_dist = spatial.distance.cosine(x.reshape(-1), y.reshape(-1))
    return cosine_dist

def get_average_err(x, y):    
    return (np.abs(x - y).sum()) / x.size

# preprocessing need to be modified
def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.ANTIALIAS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    # print(image_data.shape[0])
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data


def preprocess_func(images_folder, height, width, size_limit=10):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []
    num=0
    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        # print(image_filepath)
        image = Image.open(image_filepath)
        if len(image.split())==3:
            # print(image_filepath)
            num=num+1
             
        # print(len(image.split()))
        if len(image.split())==1 or len(image.split())==2 or len(image.split())==4:
            continue
        image_data = preprocess_image(image_filepath, height, width)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    # print("claibration_data num:",num)
    return batch_data

def _run_dequantize_linear(
    weight_tensor: np.ndarray, weight_scale: np.ndarray, weight_zp: np.ndarray, channel_axis: int
) -> Optional[np.ndarray]:
    assert weight_scale.shape == weight_zp.shape
    if weight_zp.size == 1:
        return (weight_tensor - weight_zp) * weight_scale

    assert weight_zp.ndim == 1
    reshape_dims = list(weight_tensor.shape)  # deep copy
    reshape_dims[channel_axis] = 1  # only one per channel for reshape
    channel_count = weight_tensor.shape[channel_axis]
    dequantized_weights = None
    for i in range(channel_count):
        per_channel_data = weight_tensor.take(i, channel_axis)
        dequantized_per_channel_data = (per_channel_data - weight_zp[i]) * weight_scale[i]
        if i == 0:
            dequantized_weights = np.asarray(dequantized_per_channel_data).reshape(reshape_dims)
        else:
            channel_weights = np.asarray(dequantized_per_channel_data).reshape(reshape_dims)
            dequantized_weights = np.concatenate((dequantized_weights, channel_weights), channel_axis)

    if dequantized_weights is None:
        return None

    dequantized_weights.reshape(weight_tensor.shape)
    return dequantized_weights


def create_weight_matching_QOperator(float_model_path: str, qdq_model_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Comparing weight values to help debugging accuracy loss due to quantization.

    This functions takes the float model and the qdq model, and provides a data structure for comparing
    their corresponding weights to locate quantization errors

    Arg:
        float_model_path: Path points to the float point model.
        qdq_model_path: Path points to the qdq model.

    Returns:
        Dict for comparing weight tensors. E.g.
        ```
        qdq_weight_cmp = create_weight_matching(float_model, qdq_model)
        print(qdq_weight_cmp['activation1']['float'])
        print(qdq_weight_cmp['activation1']['dequantized'])
        ```
    """
    float_onnx_model = ONNXModel(load_model_with_shape_infer(Path(float_model_path)))
    qdq_onnx_model = ONNXModel(load_model_with_shape_infer(Path(qdq_model_path)))

    matched_weights: Dict[str, Dict[str, np.ndarray]] = {}
    initializers = qdq_onnx_model.initializer()
    for node in qdq_onnx_model.nodes():
        print(node.op_type)
        # if node.op_type != DEQUANT_OP_NAME:
        #     continue  # Only care about DQ node
        weight_name: str = node.input[0]
        weight_values = find_by_name(weight_name, initializers)
        if not weight_values:
            continue  # Only care about DQ node with const inputs
        if not weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX):
            logging.error(f"Model Error in '{qdq_model_path}': Dequantized tensor name '{weight_name}' not recognized!")
            continue

        axis = -1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i

        weight_tensor = numpy_helper.to_array(weight_values)
        weight_scale = numpy_helper.to_array(find_by_name(node.input[1], initializers))
        if len(node.input) > 2:
            weight_zp = numpy_helper.to_array(find_by_name(node.input[2], initializers))
        else:
            weight_zp = np.zeros(weight_scale.shape, dtype=np.int32)

        # Perform dequantization:
        weight_quant = _run_dequantize_linear(weight_tensor, weight_scale, weight_zp, channel_axis=axis)
        weight_name = weight_name[: -len(TENSOR_NAME_QUANT_SUFFIX)]
        if weight_quant is None:
            logging.error(f"Model Error in '{qdq_model_path}': '{weight_name}' per-channel quantization on 0 channel")
            continue

        float_values = find_by_name(weight_name, float_onnx_model.initializer())
        if not float_values:
            logging.error(f"Model Error in '{float_model_path}': weight tensor '{weight_name}' not found!")
            continue
        weight_float = numpy_helper.to_array(float_values)
        matched_weights[weight_name] = {"float": weight_float, "dequantized": weight_quant}

    return matched_weights

def create_activation_matching_QOperator(
    qdq_activations: Dict[str, Sequence[np.ndarray]],
    float_activations: Optional[Dict[str, Sequence[np.ndarray]]] = None,
) -> Dict[str, Dict[str, Sequence[np.ndarray]]]:
    """Comparing activation values to help debugging accuracy loss due to quantization.

    This functions takes saved activations from the QDQ model and (optionally) the
    float point model, and provides a data structure for comparing:
        * from the qdq model, activation values before and after QDQ operation
        * across both models, activations from the orignal model vs the corresponding
          activations in the QDQ model

    Arg:
        qdq_activations: Output of `collect_activations`. This must be from a quantized
            model with QDQ format.
        float_activations: Output of `collect_activations`. This must be from the float
            point model.

    Returns:
        Dict for comparing pre and post quantized activation tensors. E.g.
        ```
        qdq_cmp = cmp_qdq_input_output(qdq_activations)
        print(qdq_cmp['activation1']['pre_qdq'][0])
        print(qdq_cmp['activation1'][`post_qdq'][0])


        qdq_cmp = cmp_qdq_input_output(qdq_activations, float_activations)
        print(qdq_cmp['activation1']['float'][0])
        print(qdq_cmp['activation1']['pre_qdq'][0])
        print(qdq_cmp['activation1'][`post_qdq'][0])
        ```
    """

    qdq_cmp: Dict[str, Dict[str, Sequence[np.ndarray]]] = {}
    for tensor_name, tensors in qdq_activations.items():
        pre_name = tensor_name
        float_tensors = float_activations.get(pre_name)
        quant_tensors = tensors
        qdq_cmp[pre_name] = {}
        qdq_cmp[pre_name]["pre_qdq"] = float_tensors
        qdq_cmp[pre_name]["post_qdq"] = quant_tensors
        # if tensor_name.endswith(QUANT_INPUT_SUFFIX):
        #     pre_name = tensor_name[: -len(QUANT_INPUT_SUFFIX)]
        #     post_qdq_tensors = qdq_activations.get(pre_name)
        #     pre_qdq_tensors = tensors
        #     _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)
        # elif tensor_name.endswith(DEQUANT_OUTPUT_SUFFIX):
        #     pre_name = tensor_name[: -len(DEQUANT_OUTPUT_SUFFIX)]
        #     pre_qdq_tensors = qdq_activations.get(pre_name)
        #     post_qdq_tensors = tensors
        #     _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)
        # elif tensor_name.endswith(_POST_QDQ_POSTFIX1):
        #     pre_name = tensor_name[: -len(_POST_QDQ_POSTFIX1)]
        #     pre_qdq_tensors = qdq_activations.get(pre_name)
        #     post_qdq_tensors = tensors
        #     _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)

    if not float_activations:
        return qdq_cmp

    for act_name, act_values in qdq_cmp.items():
        float_acts = float_activations.get(act_name)
        if float_acts is not None:
            act_values["float"] = float_acts

    return qdq_cmp

def compute_cosine_dist(
    x: Union[Sequence[np.ndarray], np.ndarray], y: Union[Sequence[np.ndarray], np.ndarray]
) -> float:

    if isinstance(x, np.ndarray):
        xlist = [x]
    else:
        xlist = x
    if isinstance(y, np.ndarray):
        ylist = [y]
    else:
        ylist = y
    if type(xlist) != list or type(ylist) != list:
        return x-y
    if len(xlist) != len(ylist):
        raise RuntimeError("Unequal number of tensors to compare!")

    left = np.concatenate(xlist).flatten()
    right = np.concatenate(ylist).flatten()

    cosine_dist = spatial.distance.cosine(left, right)

    return cosine_dist


def compute_activation_error_QOperator(
    activations_match: Dict[str, Dict[str, Sequence[np.ndarray]]],
    err_func: Callable[
        [Sequence[np.ndarray], Sequence[np.ndarray]], float
    ] = compute_cosine_dist,
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for name, match in activations_match.items():
        err_result: Dict[str, float] = {}
        err_result["qdq_err"] = err_func(match["pre_qdq"], match["post_qdq"])
        float_activation = match["float"]
        if float_activation:
            err_result["xmodel_err"] = err_func(float_activation, match["post_qdq"])
        result[name] = err_result
    return result

class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, input_info):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = None
        self.datasize = 0
        self.height = input_info[0].shape[-2]
        self.width = input_info[0].shape[-1]
        self.input_name = input_info[0].name

    def get_next(self):
        # if self.preprocess_flag:
        #     self.preprocess_flag = False
        nhwc_data_list = preprocess_func(self.image_folder, self.height, self.width, size_limit=200)
        self.datasize = len(nhwc_data_list)
        if self.enum_data_dicts == None:
            self.enum_data_dicts = iter([{self.input_name: nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)
    
    def rewind(self):
        self.enum_data_dicts = None