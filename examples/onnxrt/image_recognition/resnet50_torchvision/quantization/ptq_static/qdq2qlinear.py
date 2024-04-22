import logging
import argparse
import cv2
import numpy as np
import onnx
import re
import os
import collections
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import accuracy_score
import copy
import math
import os
import pickle
import sys
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from time import time
from typing import List
# from neural_compressor.adaptor.onnxrt import ONNXRUNTIMEAdaptor
from neural_compressor.adaptor.ox_utils.operators import OPERATORS
from neural_compressor.adaptor.ox_utils.quantizer import Quantizer
from neural_compressor import options
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.model import Model
from neural_compressor.adaptor.adaptor import adaptor_registry, Adaptor
from neural_compressor.adaptor.query import QueryBackendCapability
from neural_compressor.adaptor.ox_utils.util import PROVIDERS, ONNXRT_BACKENDS
from neural_compressor.utils.utility import LazyImport, dump_elapsed_time, \
                                            GLOBAL_STATE, MODE
from neural_compressor import quantization, PostTrainingQuantConfig
import copy
from neural_compressor.config import _Config, options
from collections import OrderedDict, defaultdict
from neural_compressor.strategy.utils.tuning_structs import OpTuningConfig
logger = logging.getLogger("evas_qdqconvertor")
from neural_compressor.algorithm import AlgorithmScheduler, ALGORITHMS
from neural_compressor.config import MixedPrecisionConfig, options
from neural_compressor.adaptor.ox_utils.util import find_by_name, dtype_to_name
from packaging.version import Version
ONNXRT152_VERSION = Version("1.5.2")
ONNXRT170_VERSION = Version("1.7.0")
ONNXRT112_VERSION = Version("1.12.0")
from copy import deepcopy
from neural_compressor.strategy.auto import AutoTuneStrategy
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.strategy.strategy import strategy_registry, TuneStrategy, STRATEGIES
from neural_compressor.strategy.utils import TuningSpace
import math
from neural_compressor.adaptor.onnxrt import ONNXRTQuery 
from importlib.util import find_spec
from neural_compressor.adaptor.ox_utils.util import quantize_data, dtype_mapping, support_pair, ValueInfo
from onnx import onnx_pb as onnx_proto
from onnx import TensorProto
from neural_compressor.adaptor.ox_utils.util import QuantizedValueType, quantize_data_per_channel
from neural_compressor.adaptor.ox_utils.util import QuantizedValue, QuantizedInitializer, \
    _get_qrange_for_qType, cast_tensor, make_quant_node, make_dquant_node
from neural_compressor.adaptor.ox_utils.operators import OPERATORS
from pdb import set_trace as bp
import neural_compressor.adaptor.ox_utils.operators.conv as conv

class Convertor:
    #Quantizer

    def __init__(self, model, q_config,fallback_list=['fp32'], reduce_range=None, add_qdq_pair_to_weight=False, optypes_to_exclude_output_quant=[],dedicated_qdq_pair=False, backend='CPUExecutionProvider'):
        # q_config, mode, static, quantization_params,
                 #op_types_to_quantize,
        """Initialization.

        Args:
            model (ModelProto or ONNXModel): onnx model or onnx model wrapper by neural compressor
            q_config (dict): op-wise quantization config.
            mode (QuantizationMode): quantizaion mode
            static (bool): static or not
            quantization_params (dict): scale and zero point of tensors
            op_types_to_quantize (list): optypes to quantize
            fallback_list (list, optional): fallback data type. Defaults to ['fp32'].
            reduce_range (bool, optional): use 7 bit or not. Defaults to None.
            add_qdq_pair_to_weight (bool, optional): add QDQ pair to weight or not. Defaults to False.
            optypes_to_exclude_output_quant (list, optional): optypes to exclude output quantization. Defaults to [].
            dedicated_qdq_pair (bool, optional): dedicate QDQ pair or not. Defaults to False.
            backend (str, optional): backend of onnxrt adaptor. Defaults to CPUExecutionProvider
        """
        self.model = ONNXModel(model) if not isinstance(model, ONNXModel) else model
        model = onnx.shape_inference.infer_shapes(self.model.model) if \
            not self.model.is_large_model else self.model.model
        self.config = q_config
        self.backend = backend
        self.mode = "QOperator"
        self.reduce_range = reduce_range
        # self.mode = mode # QuantizationMode.Value
        # self.static = static  # use static quantization for inputs.
        self.fuse_dynamic_quant = False
        # self.quantization_params = quantization_params
        # self.op_types_to_quantize = op_types_to_quantize
        self.fallback_list = fallback_list
        self.new_nodes = []
        # self.quantization_params = quantization_params
        # self.opset_version = self.check_opset_version()
        self.value_infos = {vi.name: vi for vi in model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in model.graph.output})
        self.value_infos.update({it.name: it for it in model.graph.input})
        self.replace_input = []
        self.remove_nodes = []
        # List of quantized weights
        self.quantized_value_map = {}
        self.new_value_info = {}

        # List of recalculated quantize weight for Gather op.
        self.recalculate_quantized_value = []

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        # For int8 data-type, zero point is always zero (respresented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        # if not self.static:
        #     self.op_types_to_exclude_output_quantization = op_types_to_quantize
        # else:
        self.op_types_to_exclude_output_quantization = optypes_to_exclude_output_quant

        self.add_qdq_pair_to_weight = add_qdq_pair_to_weight
        self.dedicated_qdq_pair = dedicated_qdq_pair
        
        
    def remove_redundant_pairs(self):
        """Remove redudant Q/DQ, Cast/Cast pairs."""
        self.remove_nodes = []
        self.replace_input = []
        pairs = [['QuantizeLinear', 'DequantizeLinear'], 
                 ['Cast', 'Cast'],
                 ]
        def dfs(match_nodes, node, pattern):
            if len(pattern) == 0:
                return
            start_id = 0
            end_id = len(pattern) - 1

            if node.op_type == pattern[end_id]:
                match_nodes.append(node)
            else:
                return

            if start_id == end_id:
                if all([i.op_type in ['QuantizeLinear', 'DequantizeLinear'] \
                    for i in match_nodes]):
                    pair = [str(find_by_name(i.input[2], self.model.initializer()).data_type) \
                        for i in match_nodes[::-1]]
                    if ' '.join(pair) in support_pair and support_pair[' '.join(pair)]:
                        self.replace_input.append([
                            self.model.get_children(match_nodes[1])[0],
                            match_nodes[1].output[0], 
                            match_nodes[0].input[0]])
 
                        self.remove_nodes.append(match_nodes[1])
                        if all([i.op_type in ['QuantizeLinear', 'DequantizeLinear'] \
                            for i in self.model.get_children(match_nodes[0])]) and \
                            match_nodes[0].output[0] not in self.model.output():
                            self.remove_nodes.append(match_nodes[0])
                else: # pragma: no cover
                    parent = self.model.get_parents(match_nodes[0])[0]
                    children = self.model.get_children(match_nodes[1])
                    input_dtype = '1' # float32
                    output_dtype = '1' # 'float32'
                    outs = None
                    for inp in parent.input:
                        if inp in self.new_value_info:
                            input_dtype = str(self.new_value_info[inp].new_dtype)
                            break
                    for child in children:
                        outs = [out for out in child.output if out in self.new_value_info]
                        if len(outs) > 0:
                            output_dtype = str(self.new_value_info[outs[0]].new_dtype)
                            break
                    if outs is None or len(outs) == 0 or all([not self.should_cast(i) for i in children]):
                        return
                    if input_dtype == str(match_nodes[1].attribute[0].i) and \
                        output_dtype == str(match_nodes[0].attribute[0].i) and \
                        ' '.join((output_dtype, input_dtype)) in support_pair and \
                        support_pair[' '.join((output_dtype, input_dtype))]:
                        if match_nodes[0] not in self.remove_nodes and \
                            all([i.op_type == 'Cast' and str(i.attribute[0].i) == input_dtype \
                            for i in self.model.get_children(match_nodes[0])]):
                            self.remove_nodes.append(match_nodes[0])
                        if match_nodes[1] not in self.remove_nodes:
                            self.remove_nodes.append(match_nodes[1])
                        for child in children:
                            self.replace_input.append([
                                find_by_name(child.name, self.model.model.graph.node),
                                match_nodes[1].output[0], match_nodes[0].input[0]])
                return

            children = self.model.get_children(node)
            for child in children:
                dfs(copy.deepcopy(match_nodes), child, pattern[:end_id])

        for n in self.model.nodes():
            matched = [i for i in pairs if n.op_type == i[-1]]
            if len(matched) > 0:
                for match_pair in matched:
                    visited_op = []
                    dfs(visited_op, n, match_pair)
        self.model.remove_nodes(self.remove_nodes)
        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)
        self.model.update()
        
    def quantize_inputs(self, node, indices=None, 
            initializer_use_weight_qType=True, direct_int8=False):
        """Quantize node inputs."""
        # Quantize the input
        for idx, tensor_name in enumerate(node.input):
            if indices and idx not in indices:
                continue
            initializer = find_by_name(tensor_name, self.model.initializer())
            if initializer is not None:
                if initializer.data_type != onnx_proto.TensorProto.FLOAT:
                    return
                if node.op_type not in self.op_types_to_quantize:
                    dtype = onnx_proto.TensorProto.INT8 if initializer_use_weight_qType \
                        else onnx_proto.TensorProto.UINT8
                    scheme = 'sym' if initializer_use_weight_qType else 'asym'
                else:
                    dtype = self.config[node.name]['weight']['dtype'] if \
                        initializer_use_weight_qType else \
                        self.config[node.name]['activation']['dtype']
                    scheme = self.config[node.name]['weight']['scheme'] if \
                        initializer_use_weight_qType else \
                        self.config[node.name]['activation']['scheme']
                if self.add_qdq_pair_to_weight and self.mode == 'qdq':
                    weight = self._get_quantized_weight(initializer, dtype, scheme)
                    self._update_weight(weight)
                    q_weight_name = weight.name + "_quantized"
                    zp_name = weight.name + "_zero_point"
                    scale_name = weight.name + "_scale"
                    qlinear_node = make_quant_node(tensor_name + "_QuantizeLinear",
                        [tensor_name, scale_name, zp_name], [tensor_name + "_quantized"])
                    dequant_node = make_dquant_node(tensor_name + "_DequantizeLinear",
                        [tensor_name + "_quantized", scale_name, zp_name],
                        [tensor_name + "_dequantized"])
                    self.replace_input.append([node, tensor_name, dequant_node.output[0]])
                    self.new_nodes.extend([qlinear_node, dequant_node])
                    quantized_value = QuantizedValue(weight.name, q_weight_name,
                                                     scale_name,
                                                     zp_name, 
                                                     QuantizedValueType.Initializer,
                                                     None, dtype)
                    if weight.name not in self.quantized_value_map:
                        self.quantized_value_map[weight.name] = quantized_value
                else:
                    weight = self._get_quantized_weight(initializer, dtype, scheme)
                    self._update_weight(weight)
                    q_weight_name = weight.name + "_quantized"
                    zp_name = weight.name + "_zero_point"
                    scale_name = weight.name + "_scale"
 
                    inputs = [q_weight_name, scale_name, zp_name]
                    output_name = tensor_name + '_DequantizeLinear'
                    dequant_node = onnx.helper.make_node("DequantizeLinear", inputs,
                        [tensor_name + '_dequantized'], tensor_name + '_DequantizeLinear')
                    self.new_nodes.append(dequant_node)
                    self.replace_input.append([node, tensor_name, dequant_node.output[0]])
                    quantized_value = QuantizedValue(weight.name, q_weight_name,
                                                     scale_name,
                                                     zp_name, 
                                                     QuantizedValueType.Initializer,
                                                     None, dtype)
                    if weight.name not in self.quantized_value_map:
                        self.quantized_value_map[weight.name] = quantized_value
            else:
                if tensor_name in self.value_infos and \
                    self.value_infos[tensor_name].type.HasField('tensor_type') and \
                    self.value_infos[tensor_name].type.tensor_type.elem_type != TensorProto.FLOAT:
                    return

                if tensor_name in self.quantized_value_map:
                    scale_name = self.quantized_value_map[tensor_name].scale_name
                    zp_name = self.quantized_value_map[tensor_name].zp_name
                    data_found = True
                else:
                    data_found, scale_name, zp_name, _, _ = \
                        self._get_quantization_params(tensor_name)
 
                if self.config[node.name.split('_quant')[0]]['activation']['quant_mode'] != \
                    'dynamic':
                    if data_found == False:
                        raise ValueError(
                            "Quantization parameters are not specified for param {}."
                            "In static mode quantization params for inputs and outputs "
                            "of nodes to be quantized are required.".format(tensor_name))
                    if direct_int8:
                        if node.input[0] not in self.quantized_value_map:
                            return
                    q_input = tensor_name
                    q_output = tensor_name + "_" + node.name + "_QuantizeLinear" if \
                        tensor_name not in self.model.input() else tensor_name + "_quantized"
                    dq_input = q_output
                    dq_output = tensor_name + "_" + node.name + "_dequantized" if \
                        tensor_name not in self.model.input() else tensor_name + "_dequantized"
                    self.replace_input.append([node, tensor_name, dq_output])
                    if tensor_name in self.model.input() and tensor_name in self.quantized_value_map:
                        continue

                    quant_node_name = tensor_name + "_" + node.name + "_QuantizeLinear"
                    dequant_node_name = tensor_name + "_" + node.name + "_DequantizeLinear"
                    qlinear_node = make_quant_node(
                        quant_node_name, [q_input, scale_name, zp_name], [q_output])
                    dequant_node = make_dquant_node(
                        dequant_node_name, [dq_input, scale_name, zp_name], [dq_output])
                    self.new_nodes.extend([qlinear_node, dequant_node])
                    quantized_value = QuantizedValue(tensor_name, dq_output,
                                                     scale_name,
                                                     zp_name, 
                                                     QuantizedValueType.Input)
                    if tensor_name not in self.quantized_value_map:
                        self.quantized_value_map[tensor_name] = quantized_value
                else:
                    qlinear_node = self.model.find_node_by_name(tensor_name + "_QuantizeLinear",
                                                                self.new_nodes,
                                                                self.model.graph())
                    if qlinear_node is None:
                        if self.fuse_dynamic_quant and \
                            self.config[node.name]['activation']['dtype'] == \
                                onnx_proto.TensorProto.UINT8 and \
                            self.config[node.name]['activation']['scheme'] == 'asym':
                            scale_name = tensor_name + "_scale"
                            zeropoint_name = tensor_name + "_zero_point"
                            if find_by_name(scale_name, self.model.initializer()):
                                self.model.remove_initializer(
                                    find_by_name(scale_name, self.model.initializer()))
                            if find_by_name(zeropoint_name, self.model.initializer()):
                                self.model.remove_initializer(
                                    find_by_name(zeropoint_name, self.model.initializer()))
                            qlinear_node = onnx.helper.make_node("DynamicQuantizeLinear", 
                                [tensor_name],
                                [tensor_name + "_quantized", scale_name, zeropoint_name],
                                tensor_name + "_QuantizeLinear")
                        else:
                            scale_name, zp_name, _, _ = \
                                self._get_dynamic_input_quantization_params(
                                tensor_name, self.config[node.name]['activation']['dtype'])
                            qlinear_node = make_quant_node(tensor_name + "_QuantizeLinear",
                                                        [tensor_name, scale_name, zp_name], 
                                                        [tensor_name + "_quantized"])
                        if qlinear_node not in self.new_nodes:
                            self.new_nodes.append(qlinear_node)
                        self.quantized_value_map[tensor_name] = QuantizedValue(
                            tensor_name, 
                            qlinear_node.output[0],
                            scale_name, 
                            zp_name, 
                            self.config[node.name]['activation']['dtype'])                        
                    self.replace_input.append([node, tensor_name, qlinear_node.output[0]])
 


    def should_convert(self, node):
        """Check if node should be converted."""
        # name = node.name.split('_quant')[0]
        name = node.name.split('_quant')[0]
        if name in self.config and self.config[name] not in self.fallback_list and \
            (self.config[name]['activation']['quant_mode'] == 'dynamic' or self.mode != 'qdq'):
            return True
        else:
            return False
        
    def _get_quantization_params(self, param_name):
        """Create initializers and inputs in the graph for zero point and scale of output.

        Zero point and scale values are obtained from self.quantization_params if specified.

        Args:
            param_name (string): Name of the quantization parameter.

        """
        if self.quantization_params is None or param_name not in self.quantization_params:
            return False, "", "", "", ""

        params = self.quantization_params[param_name]
        if params is None or len(params) != 2:
            raise ValueError("Quantization parameters should contain zero point and scale. "
                             "Specified values for output {}: {}".format(param_name, params))

        zero_point_values = [params[0]]
        zero_point_shape = []
        zero_point_name = param_name + "_zero_point"
        zero_point_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[params[0].dtype]

        scale_values = [params[1]]
        scale_shape = []
        scale_name = param_name + "_scale"

        # Add initializers
        init_zp = onnx.helper.make_tensor(zero_point_name, zero_point_type, 
                                          zero_point_shape, zero_point_values)
        self.model.add_initializer(init_zp)
        init_scale = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, 
                                             scale_shape, scale_values)
        self.model.add_initializer(init_scale)

        return True, scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_quantized_weight(self, initializer, qType, scheme):
        """Get quantized weight."""
        if initializer.name in self.quantized_value_map:
            return self.quantized_value_map[initializer.name]
        weights_data = self.tensor_proto_to_array(initializer, os.path.dirname(self.model.model_path)) if \
            self.model.model_path is not None else self.tensor_proto_to_array(initializer)
        rmin, rmax, zero_point, scale, quantized_weights_data = quantize_data(
            weights_data.flatten().tolist(), _get_qrange_for_qType(qType, \
            self.reduce_range), qType, scheme)
        weight = QuantizedInitializer(initializer.name,
                                      initializer, [rmin], [rmax], [zero_point], [scale],
                                      weights_data,
                                      quantized_weights_data,
                                      axis=None,
                                      qType=qType)

        return weight
        
        
    
    def convert_qdq_to_operator_oriented(self):
        
        """Convert QDQ to QOperator format."""
        self.new_nodes = []
        self.remove_nodes = []
        self.replace_input = []
        for node in self.model.nodes(): 
            # make all nodes end with _quant ,aim to convert QOperator
            if node.op_type not in ['QuantizeLinear', 'DequantizeLinear']:
                node.name = node.name + "_quant"
            
            if node.op_type not in ['QuantizeLinear', 'DequantizeLinear'] and \
                self.should_convert(node):
                op_converter = OPERATORS[node.op_type](self, node)
                mode = self.config[node.name.split('_quant')[0]]['activation']['quant_mode']
                if op_converter.convert_check(mode):
                    op_converter.convert(mode)
                
                
                
            
                
        self.model.graph().node.extend(self.new_nodes)
        self.model.remove_nodes(self.remove_nodes)
        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)
        self.model.update()
        
    def merge_dedicated_qdq_pair(self):
        """Merge dedicated Q/DQ pairs."""
        self.remove_nodes = []
        self.replace_input = []
        self.new_nodes = []
        if self.mode == 'qdq' and self.dedicated_qdq_pair:
            for node in self.model.nodes():
                if node.op_type in ['QuantizeLinear']:
                    children = self.model.get_children(node)
                    if len([i for i in children if i.op_type in ['DequantizeLinear']]) < 2:
                        continue
                    for idx, child in enumerate(children):
                        if child.op_type not in ['DequantizeLinear']:
                            continue
                        if self.should_quantize(self.model.get_children(child)[0]):
                            inputs = [self.model.get_parents(node)[0].output[0],
                                      node.input[1], node.input[2]]
                            self.new_nodes.append(onnx.helper.make_node(
                                "QuantizeLinear",
                                inputs,
                                [node.output[0] + '_' + str(idx)],
                                node.name + '_' + str(idx)))
                            self.replace_input.append([child, node.output[0], 
                                                node.output[0] + '_' + str(idx)])
                        else:
                            self.remove_nodes.append(child)
                            self.replace_input.append([self.model.get_children(child)[0],
                                                child.output[0], node.input[0]])
                    self.remove_nodes.append(node)
            self.model.remove_nodes(self.remove_nodes)
            self.model.graph().node.extend(self.new_nodes)
            for node, old_input_name, new_input_name in self.replace_input:
                self.model.replace_node_input(node, old_input_name, new_input_name)
            self.model.update()
        elif self.mode != 'qdq' or not self.dedicated_qdq_pair:
            target_type = ['QuantizeLinear', 'DequantizeLinear']
            for op_type in target_type:
                for node in self.model.nodes():
                    children = self.model.get_children(node)
                    dq_nodes = [i for i in children if i.op_type == op_type]
                    if len(dq_nodes) < 2 or node.op_type in ['Split']:
                        continue
                    datas = []
                    for n in dq_nodes:
                        datas.append([onnx.numpy_helper.to_array(
                                          find_by_name(n.input[1], self.model.initializer())), 
                                      onnx.numpy_helper.to_array(
                                          find_by_name(n.input[2], self.model.initializer()))])
                    for idx, data in enumerate(datas):
                        repeaded_id = [i for i, item in enumerate(datas[idx:]) if item == data]
                        for i in repeaded_id[1:]:
                            self.remove_nodes.append(dq_nodes[i])
                            self.replace_input.append([self.model.get_children(dq_nodes[i])[0],
                                                       dq_nodes[i].output[0], 
                                                       dq_nodes[idx].output[0]])
                self.model.remove_nodes(self.remove_nodes)
                self.model.graph().node.extend(self.new_nodes)
                for node, old_input_name, new_input_name in self.replace_input:
                    self.model.replace_node_input(node, old_input_name, new_input_name)
                self.model.update()
        
        if self.mode == 'qdq':
            for node in self.model.nodes():
                if node.op_type in ['QuantizeLinear'] and len(self.model.get_parents(node)) > 0:
                    if 'QuantizeLinear' in [sibling.op_type \
                                            for sibling in self.model.get_siblings(node)]:
                        continue
                    for sibling in self.model.get_siblings(node):
                        if not self.should_quantize(sibling) and sibling.op_type in OPERATORS:
                            for inp_idx in range(len(sibling.input)):
                                if sibling.input[inp_idx] == node.input[0]:
                                    self.replace_input.append([sibling,
                                        sibling.input[inp_idx],
                                        self.model.get_children(node)[0].output[0]])
            for node, old_input_name, new_input_name in self.replace_input:
                self.model.replace_node_input(node, old_input_name, new_input_name)
            self.model.update()
            
    def should_quantize(self, node):
        """Check if node should be quantized."""
        if node.name in self.config and self.config[node.name] not in self.fallback_list:
            return True
        elif node.name.split('_quant')[0] in self.config and \
            self.config[node.name.split('_quant')[0]] not in self.fallback_list:
            return True
        else:
            return False
        
    def should_cast(self, node):
        """Check if node should be casted."""
        if node.name in self.config and self.config[node.name] != 'fp32': # pragma: no cover
            return True
        else:
            return False
            
            
    # def insert_qdq(self):
    #     """Insert Q/DQ pairs."""
    #     for node in self.model.nodes():
    #         print(node.name)
    #         if self.should_quantize(node):
    #             op_quantizer = OPERATORS[node.op_type](self, node)
    #             if op_quantizer.quantize_check():
    #                 op_quantizer.quantize = conv.only_quantize_op
    #                 op_quantizer.quantize()
    #         elif self.should_cast(node): # pragma: no cover
              
    #             op_caster = OPERATORS[node.op_type](self, node)
    #             op_caster.cast()

    #     self.model.graph().node.extend(self.new_nodes)
    #     self.model.remove_nodes(self.remove_nodes)

    #     for node, old_input_name, new_input_name in self.replace_input:
    #         self.model.replace_node_input(node, old_input_name, new_input_name)
    #     self.model.update()

        
    def convert_model(self):
        """Quantize onnx qdq model."""
        
        
        # self.insert_qdq()
        # for node in self.model.nodes():
        #     print(node.name)
   
        
  
        # self.remove_redundant_pairs()
        # onnx.save(self.model.model,"tempremove.onnx")
       

        self.convert_qdq_to_operator_oriented()
        onnx.save(self.model.model,"tempqop.onnx")
        bp()
        # for node in self.model.nodes():
        #     print(node.name)
        #     print("########################")
        self.merge_dedicated_qdq_pair() 
 
        self.model.remove_unused_constant()

        

        return self.model.model
        
class EONNXRUNTIMEAdaptor(Adaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super().__init__(framework_specific_info)
        self.__config_dict = {}
        self.quantizable_ops = []
        self.device = framework_specific_info["device"]
        self.static = framework_specific_info["approach"] == "post_training_static_quant"
        self.dynamic = framework_specific_info["approach"] == "post_training_dynamic_quant"
        self.domain = framework_specific_info.get("domain", "auto")
        self.recipes = framework_specific_info.get("recipes", {})
        self.recipes['add_qdq_pair_to_weight'] = True
        self.backend = PROVIDERS[framework_specific_info["backend"]]
        self.performance_only = framework_specific_info.get("performance_only", False)

        if self.backend not in ort.get_all_providers():
            logger.warning("{} backend is not supported in current environment, "
                "supported backends: {}".format(ONNXRT_BACKENDS[self.backend],
                [ONNXRT_BACKENDS[i] for i in ort.get_all_providers() if i in ONNXRT_BACKENDS]))

        # get quantization format according to framework_specific_info
        if (not self.dynamic and "format" in framework_specific_info and \
            framework_specific_info["format"].lower() == 'qdq') or \
            self.backend == 'TensorrtExecutionProvider':
            self.format = "qdq"
        else:
            if not self.dynamic:
                self.format = "qlinearops"
            else:
                self.format = "integerops"
                if "format" in framework_specific_info and \
                    framework_specific_info["format"].lower() == 'qdq':
                    logger.warning("Dynamic approach doesn't support QDQ format.")
    
        # get quantization config file according to backend
        config_file = None
        if self.backend == 'CPUExecutionProvider':
            config_file = 'onnxrt.yaml'
        elif self.backend == 'TensorrtExecutionProvider':
            config_file = 'onnxrt_trt.yaml'
        elif self.backend == 'CUDAExecutionProvider':
            config_file = 'onnxrt_cuda.yaml'
        else: # pragma: no cover
            assert False, "{} provider is not supported in current environment, " \
                "supported providers: {}".format(self.backend,
                [provider for provider in PROVIDERS.values()])

        self.query_handler_ext = None
        self.work_space = framework_specific_info["workspace_path"]
        os.makedirs(self.work_space, exist_ok=True)
        if framework_specific_info["approach"] == 'post_training_auto_quant' and \
            self.format != "integerops":
            # if approach is post_training_auto_quant, 
            # both static and dynamic quantization will be performed
            self.query_handler = ONNXRTQuery(
                static=True, 
                format=self.format,
                local_config_file=os.path.join(os.path.dirname(__file__), config_file))
            self.query_handler_ext = ONNXRTQuery(
                dynamic=True, 
                format=self.format,
                local_config_file=os.path.join(os.path.dirname(__file__), config_file))
        else:
            self.query_handler = ONNXRTQuery(
                dynamic=self.dynamic, 
                static=self.static, 
                format=self.format,
                # local_config_file=os.path.join(os.path.dirname(__file__), config_file))
                local_config_file= "/home/wrq/anaconda3/envs/quant/lib/python3.9/site-packages/neural_compressor/adaptor/onnxrt.yaml"
            )

        self.quantizable_op_types = []

        for precision in self.query_handler.get_precisions():
            if precision != 'fp32':
                if self.device == 'cpu' and precision == 'fp16':
                    continue
                self.quantizable_op_types += \
                    self.query_handler.get_op_types_by_precision(precision=precision)

        if self.backend == 'TensorrtExecutionProvider':
            self.recipes['add_qdq_pair_to_weight'] = True
            self.recipes['dedicated_qdq_pair'] = True
            self.recipes['graph_optimization_level'] = 'DISABLE_ALL'
            self.recipes['optypes_to_exclude_output_quant'] = ['Conv', 'Gemm', 'Add', 'MatMul']
            self.static = True
            self.dynamic = False

        self.evaluate_nums = 0

        self.fp32_results = []
        self.fp32_preds_as_label = False
        self.quantize_config = {} # adaptor should know current configs at any time
        self.quantize_params = {} # adaptor should know current params at any time
        self.min_max = None
        self.pre_optimized_model = None
        self.optype_statistics = None
        
    
    def query_fw_capability(self, model):
        
        
        """The function is used to query framework capability.
        TODO: will be replaced by framework query API

        Args:
            model: onnx model

        Returns:
            (dict): quantization capability
        """
        # optype_wise and op_wise capability
        
        self._pre_optimize(model)
        onnx = LazyImport('onnx')
        onnx.save(model.model,"pre_optimize.onnx")
      
        recipes_ops = {}
        recipes_ops['first_conv_or_matmul_quantization'] = []
        recipes_ops['last_conv_or_matmul_quantization'] = []
        recipes_ops['pre_post_process_quantization'] = []
        exclude_first_quantizable_op = True if 'first_conv_or_matmul_quantization' in \
            self.recipes and not self.recipes['first_conv_or_matmul_quantization'] \
            else False
        exclude_last_quantizable_op = True if 'last_conv_or_matmul_quantization' in \
            self.recipes and not self.recipes['last_conv_or_matmul_quantization'] \
            else False
        exclude_pre_post_process = True if 'pre_post_process_quantization' in \
            self.recipes and not self.recipes['pre_post_process_quantization'] \
            else False

        quantizable_optype = set([i.op_type for i in self.pre_optimized_model.nodes()])
        optype_wise = OrderedDict()
        op_wise = OrderedDict()
        for query in [self.query_handler, self.query_handler_ext]:
            if query is None:
                continue
            precisions = query.get_precisions()

            for precision in precisions:
                if precision in ['fp16', 'bf16'] and (self.device == 'cpu' or self.backend != 'CUDAExecutionProvider'):
                    continue
                elif precision == 'bf16' and 'CUDAExecutionProvider' not in ort.get_available_providers():
                    continue
                # get supported optype for target precision
                optypes = query.get_op_types_by_precision(precision) if \
                    query.get_op_types_by_precision(precision) != ['*'] else \
                    optype_wise.keys()

                configs = query.get_quantization_capability()[precision] if \
                    precision in query.get_quantization_capability() else \
                    {'default': {'weight': {'dtype': precision}, 'activation': {'dtype': precision}}}

                if self.backend == 'TensorrtExecutionProvider' and \
                    precision not in query.get_fallback_list():
                    optypes.append('Add')

                for op in optypes:
                    if op not in quantizable_optype:
                        continue
                    if op not in configs:
                        if 'default' in configs:
                            op_capability = copy.deepcopy(configs['default'])
                        else:
                            continue
                    else:
                        op_capability = copy.deepcopy(configs[op])

                    if precision in ['int8', 'uint8']:
                        if self.static:
                            op_capability['activation']['quant_mode'] = 'static'
                        elif self.dynamic:
                            op_capability['activation']['quant_mode'] = 'dynamic'
                        elif query == self.query_handler: # query static capability for auto
                            op_capability['activation']['quant_mode'] = 'static'
                        elif query == self.query_handler_ext: # query dynamic capability for auto
                            op_capability['activation']['quant_mode'] = 'dynamic'

                    if op not in optype_wise.keys():
                        optype_wise[op] = [op_capability]
                    elif op_capability not in optype_wise[op]:
                        optype_wise[op].append(op_capability)

        if self.format == "qdq":
            self._optypewise_filter_for_qdq(optype_wise)

        first_quantizable_node = []
        last_quantizable_node = []
        all_conv_matmul = []
        attention_matmul = []
        for _, node in enumerate(self.pre_optimized_model.nodes()):
            if node.op_type in ['Conv', 'MatMul', 'Attention']:
                # get first Conv or MatMul node
                if len(first_quantizable_node) == 0:
                    first_quantizable_node.append(node)

                # get last Conv or MatMul node
                if len(last_quantizable_node) != 0:
                    last_quantizable_node.pop()
                last_quantizable_node.append(node)

                all_conv_matmul.append(node)
                if node.op_type != 'Conv':
                    attention_matmul.append(node)
        
        if len(first_quantizable_node) != 0:
            recipes_ops['first_conv_or_matmul_quantization'] = [(first_quantizable_node[0].name, 
                                                                first_quantizable_node[0].op_type)]
        if len(last_quantizable_node) != 0:
            recipes_ops['last_conv_or_matmul_quantization'] = [(last_quantizable_node[0].name, 
                                                                last_quantizable_node[0].op_type)]
        
        
        ffn_matmul = []
        attention_matmul_optype = [node.op_type for node in attention_matmul]
        # find matmul ops in feed forward network (FFN) structure whitch mainly in transfomers based NLP models
        if len(attention_matmul) > 0 and 'Attention' in attention_matmul_optype:
            # model is optimized and Attention is fused,
            # index of Attention is used as split to find FFN MatMul
            first_attention_index = attention_matmul_optype.index('Attention')
            attention_matmul_optype = attention_matmul_optype[first_attention_index:]
            attention_matmul = attention_matmul[first_attention_index:]
            attention_index = list(np.where(np.array(attention_matmul_optype) == 'Attention')[0])
            block_len = attention_index[1] - attention_index[0] if len(attention_index) > 2 else 4
            for idx in range(len(attention_index)):
                if idx != len(attention_index) - 1:
                    index = attention_index[idx + 1]
                    if index - 2 >= 0 and index - 1 >= 0:
                        ffn_matmul.append([attention_matmul[index - 2], 
                                        attention_matmul[index - 1]])
                else:
                    index = attention_index[idx]
                    if index + block_len - 2 < len(attention_matmul) and \
                        index + block_len - 1 < len(attention_matmul):
                        ffn_matmul.append([attention_matmul[index + block_len - 2], 
                                        attention_matmul[index + block_len - 1]])
        else:
            # model is not optimized or Attention isn't fused, 
            # query MatMul, key MatMul and value MatMul are used as split to find FFN MatMul
            qkv = self.pre_optimized_model.find_qkv_in_attention(find_all=True)
            if len(qkv) != 0:
                attention_starts = [nodes[0] for nodes in qkv]
                attention_index = [np.where(np.array([n.name for n in attention_matmul]) \
                                            == attention_start)[0].tolist()[0] \
                                                for attention_start in attention_starts]
                block_len = attention_index[1] - attention_index[0] if len(attention_index) > 2 else 4
                for idx in range(len(attention_index)):
                    if idx != len(attention_index) - 1:
                        index = attention_index[idx + 1]
                        if index - 2 >= 0 and index - 1 >= 0:
                            ffn_matmul.append([attention_matmul[index - 2],
                                            attention_matmul[index - 1]])
                    else:
                        index = attention_index[idx]
                        if index + block_len - 2 < len(attention_matmul) and \
                            index + block_len - 1 < len(attention_matmul):
                            ffn_matmul.append([attention_matmul[index + block_len - 2],
                                            attention_matmul[index + block_len - 1]])

        block_wise = []
        for block in reversed(ffn_matmul):
            node_info = []
            for node in block:
                node_info.append((node.name, node.op_type))
            if len(node_info) != 0:
                block_wise.append(node_info)

        for _, node in enumerate(self.pre_optimized_model.nodes()):
            # for TRT EP, only insert Q/DQ to inputs of Add nodes followed by ReduceMean
            if node.op_type == 'Add' and self.backend == 'TensorrtExecutionProvider':
                children = self.pre_optimized_model.get_children(node)
                if 'ReduceMean' not in [i.op_type for i in children]:
                    op_wise.update({(node.name, node.op_type): 
                        [{'weight': {'dtype': 'fp32'}, 'activation': {'dtype': 'fp32'}}]})
                    continue

            if node.op_type in optype_wise:
                if (exclude_first_quantizable_op and node in first_quantizable_node) \
                    or (exclude_last_quantizable_op and node in last_quantizable_node):
                    tmp_cfg = copy.deepcopy(optype_wise[node.op_type])
                    tmp_cfg = list(filter(lambda x:'quant_mode' not in x['activation'], tmp_cfg))
                    op_wise.update({(node.name, node.op_type): tmp_cfg})
                    continue
                op_wise.update(
                    {(node.name, node.op_type): copy.deepcopy(optype_wise[node.op_type])})

        # only when first and last quantizable nodes are found and they are not the same,
        # fallback pre/postprocess ops
        if len(first_quantizable_node) != 0 and \
        len(last_quantizable_node) != 0 and \
        first_quantizable_node[0].name != last_quantizable_node[0].name:
            # get backbone nodes
            from collections import deque
            
            # get nodes between first quantizable node and last quantizable node
            backbone_queue = deque(last_quantizable_node)
            backbone_nodes = self.pre_optimized_model.get_nodes_chain(backbone_queue, first_quantizable_node)

            # get extra Conv or MatMul nodes not between first quantizable node and last quantizable node
            backbone_queue_extra = deque()
            for conv_or_matmul in all_conv_matmul:
                if conv_or_matmul.name not in backbone_nodes:
                    backbone_queue_extra.append(conv_or_matmul)
                    backbone_nodes = self.pre_optimized_model.get_nodes_chain(backbone_queue_extra, 
                                                    first_quantizable_node, backbone_nodes)
            backbone_nodes += [i.name for i in first_quantizable_node]
            
            for _, node in enumerate(self.pre_optimized_model.nodes()):
                if node.name not in backbone_nodes and node.op_type in optype_wise:
                    recipes_ops['pre_post_process_quantization'].append((node.name, node.op_type))
            if exclude_pre_post_process:
                for _, node in enumerate(self.pre_optimized_model.nodes()):
                    if node.op_type in optype_wise:
                        # nodes not in backbone are not quantized
                        if node.name not in backbone_nodes:
                            tmp_cfg = copy.deepcopy(optype_wise[node.op_type])
                            tmp_cfg = list(filter(lambda x:'quant_mode' not in x['activation'], tmp_cfg))
                            op_wise.update({(node.name, node.op_type): tmp_cfg})
                            continue
                        if (node.name, node.op_type) in op_wise:
                            op_wise.update(
                                {(node.name, node.op_type): copy.deepcopy(op_wise[(node.name, node.op_type)])})
                        else: # pragma: no cover
                            op_wise.update(
                                {(node.name, node.op_type): copy.deepcopy(optype_wise[node.op_type])})

        return {'optypewise': optype_wise, 'opwise': op_wise, 'recipes_ops': recipes_ops, 'block_wise': block_wise}


            
    def quantize_config_generate(self,tune_cfg,model):
        from neural_compressor.adaptor.ox_utils.util import QuantizationMode
        ort_version = Version(ort.__version__)

        if self.format == "qlinearops":
            format = QuantizationMode.QLinearOps
        elif self.format == "qdq":
            assert ort_version >= ONNXRT170_VERSION, 'QDQ mode needs onnxruntime1.7.0 or newer'
            format = "qdq"
        else:
            format = QuantizationMode.IntegerOps

        self.quantizable_ops = self._query_quantizable_ops(model.model)
        quantize_config = self._cfg_to_quantize_config(tune_cfg)
        return quantize_config
            
    def _cfg_to_quantize_config(self, tune_cfg):
        quantize_config = {}
        quantize_config['calib_iteration'] = tune_cfg['calib_iteration']
        granularity = 'per_tensor'
        algorithm = 'minmax'

        from onnx import onnx_pb as onnx_proto
        for _, op in enumerate(self.quantizable_ops):
            if (op.name, op.op_type) not in tune_cfg['op']:
                continue
            if tune_cfg['op'][(op.name, op.op_type)]['activation']['dtype'] in \
                self.query_handler.get_fallback_list():
                quantize_config[op.name] = \
                    tune_cfg['op'][(op.name, op.op_type)]['activation']['dtype']
            else:
                node_config = copy.deepcopy(tune_cfg['op'][(op.name, op.op_type)])
                for tensor, config in tune_cfg['op'][(op.name, op.op_type)].items():
                    if 'granularity' not in config:
                        node_config[tensor]['granularity'] = granularity
                    if 'algorithm' not in config:
                        node_config[tensor]['algorithm'] = algorithm
                    if config['dtype'] == "int8":
                        node_config[tensor]['dtype'] = onnx_proto.TensorProto.INT8
                        if 'scheme' not in config:
                            node_config[tensor]['scheme'] = 'sym'
                    else:
                        node_config[tensor]['dtype'] = onnx_proto.TensorProto.UINT8
                        if 'scheme' not in config:
                            node_config[tensor]['scheme'] = 'asym'
                quantize_config[op.name] = node_config

        return quantize_config

    def _query_quantizable_ops(self, model):
        for node in model.graph.node:
            if node.op_type in self.quantizable_op_types and node not in self.quantizable_ops:
                self.quantizable_ops.append(node)

        return self.quantizable_ops
    
    def _query_quantizable_op_types(self):
        quantizable_op_types = self.query_handler.get_op_types_by_precision(precision='int8')
        return quantizable_op_types
    
    
    def _pre_optimize(self, model, level=1):
        from neural_compressor import options
        from neural_compressor.adaptor.ox_utils.util import \
            remove_init_from_model_input, split_shared_bias
        remove_init_from_model_input(model)
        sess_options = ort.SessionOptions()
        optimization_levels = {
                'DISABLE_ALL': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                'ENABLE_BASIC': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                'ENABLE_EXTENDED': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                'ENABLE_ALL': ort.GraphOptimizationLevel.ORT_ENABLE_ALL}
     
        if not isinstance(self.query_handler.get_graph_optimization(), list):
            level = self.query_handler.get_graph_optimization()
        elif options.onnxrt.graph_optimization.level is not None:
            level = options.onnxrt.graph_optimization.level
        elif self.recipes.get('graph_optimization_level', None) is not None:
            level = self.recipes['graph_optimization_level']
        else:
            if self.domain == "auto" and self._detect_domain(model):
                self.domain = 'nlp' 
            level = 'ENABLE_EXTENDED' if self.domain == 'nlp' else 'ENABLE_BASIC'
            logger.warning("Graph optimization level is automatically set to {}. "
                "You can use 'recipe' argument in 'PostTrainingQuantConfig'" 
                "to overwrite it".format(level))
        sess_options.graph_optimization_level = optimization_levels[level]
        sess_options.optimized_model_filepath = os.path.join(self.work_space, \
            "Optimized_model.onnx")
        if sys.version_info < (3,10) and find_spec('onnxruntime_extensions'): # pragma: no cover
            from onnxruntime_extensions import get_library_path
            sess_options.register_custom_ops_library(get_library_path())
        backend = self.backend if self.backend != 'TensorrtExecutionProvider' else 'CUDAExecutionProvider'
        if not model.is_large_model:
            ort.InferenceSession(model.model.SerializeToString(),
                                 sess_options,
                                 providers=[backend])
        elif model.model_path is not None: # pragma: no cover
            ort.InferenceSession(model.model_path,
                                 sess_options,
                                 providers=[backend])
        else: # pragma: no cover 
            logger.warning('Please use model path instead of onnx model object to quantize')

        tmp_model = onnx.load(sess_options.optimized_model_filepath, load_external_data=False)
        if model.is_large_model: # pragma: no cover
            from onnx.external_data_helper import load_external_data_for_model
            load_external_data_for_model(tmp_model, os.path.split(model.model_path)[0])
        model.model_path = sess_options.optimized_model_filepath
        # model.model = self._replace_gemm_with_matmul(tmp_model).model if \
        #     options.onnxrt.graph_optimization.gemm2matmul and self.recipes.get('gemm_to_matmul', True) else \
        #     tmp_model
        model.model = tmp_model
        model.model = self._rename_node(model.model)
        model = self._revert_fusedconv(model)
        if self.backend == 'TensorrtExecutionProvider':
            model = self._revert_conv_add_fusion(model)
        model = split_shared_bias(model)
        model.topological_sort()
        
        self.pre_optimized_model = model
        
        
    def _detect_domain(self, model):
        """Automatically detect whether the model belongs to NLP domain.

        Args:
            model (ONNXModel): ONNXModel wrapped model

        Returns:
            bool: the model belongs to NLP domain or not
        """
        is_nlp = False
        # 1. according to initializer names
        initializer_names = [init.name for init in model.model.graph.initializer]
        pattern = ".*word.*embedding.*"
        for name in initializer_names:
            obj = re.findall(pattern, name)
            if len(obj) > 0:
                is_nlp = True
                break
        
        # 2. according to input
        # typically, NLP models have multiple inputs, 
        # and the dimension of each input is usually 2 (batch_size, max_seq_len)
        if not model.is_large_model:
            sess = ort.InferenceSession(model.model.SerializeToString(), providers=[self.backend])
        elif model.model_path is not None: # pragma: no cover
            sess = ort.InferenceSession(model.model_path, providers=[self.backend])
        else: # pragma: no cover
            assert False, "Please use model path instead of onnx model object to quantize."
        input_shape_lens = [len(input.shape) for input in  sess.get_inputs()]
        if len(input_shape_lens) > 1 and all(shape_len == 2 for shape_len in input_shape_lens):
            is_nlp = True

        # 3. according to attention structure
        qkv = model.find_qkv_in_attention()
        if len(qkv) != 0:
            is_nlp = True

        # 4. according to LSTM/Attention optype
        op_types = [node.op_type for node in model.model.graph.node]
        if "LSTM" in op_types or 'Attention' in op_types:
            is_nlp = True

        logger.warning("The model is automatically detected as {} model. "
            "You can use 'domain' argument in 'PostTrainingQuantConfig' "
            "to overwrite it".format("an NLP" if is_nlp else "a non-NLP"))
        return is_nlp
   
   
    @staticmethod
    def _replace_gemm_with_matmul(model):
        new_nodes = []
        from onnx import numpy_helper
        from neural_compressor.model.onnx_model import ONNXModel
        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)

        for node in model.nodes():
            if node.op_type == 'Gemm':
                alpha = 1.0
                beta = 1.0
                transA = 0
                transB = 0
                for attr in node.attribute:
                    if attr.name == 'alpha':
                        alpha = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'beta':
                        beta = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transA':
                        transA = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transB':
                        transB = onnx.helper.get_attribute_value(attr)
                if alpha == 1.0 and beta == 1.0 and transA == 0:
                    inputB = node.input[1]
                    if transB == 1:
                        B = model.get_initializer(node.input[1])
                        if B:
                            # assume B is not used by any other node
                            B_array = numpy_helper.to_array(B)
                            B_trans = numpy_helper.from_array(B_array.T)
                            B_trans.name = B.name
                            model.remove_initializer(B)
                            model.add_initializer(B_trans)

                            #TBD this is for onnx model zoo, which are all in old IR version
                            if model.model.ir_version < 4:
                                for input in model.model.graph.input:
                                    if input.name == B_trans.name:
                                        for i, dim in enumerate(input.type.tensor_type.shape.dim):
                                            dim.dim_value = B_array.T.shape[i]

                        else:
                            inputB += '_Transposed'
                            transpose_node = onnx.helper.make_node('Transpose',
                                                                inputs=[node.input[1]],
                                                                outputs=[inputB],
                                                                name=node.name+'_Transpose')
                            new_nodes.append(transpose_node)

                    matmul_node = onnx.helper.make_node('MatMul',
                            inputs=[node.input[0], inputB],
                            outputs=[node.output[0] + ('_MatMul' if len(node.input)>2 else '')],
                            name=node.name + '_MatMul')
                    new_nodes.append(matmul_node)

                    if len(node.input) > 2:
                        add_node = onnx.helper.make_node('Add',
                            inputs=[node.output[0] + '_MatMul', node.input[2]],
                            outputs=node.output,
                            name=node.name + '_Add')
                        new_nodes.append(add_node)

                # unsupported
                else:
                    new_nodes.append(node)

            # not GEMM
            else:
                new_nodes.append(node)

        model.graph().ClearField('node')
        model.graph().node.extend(new_nodes)

        return model
         
    def _rename_node(self, model):
        node_names = [i.name for i in model.graph.node]
        if len(set(node_names)) < len(node_names):
            logger.warning("This model has nodes with the same name, please check" \
                "renamed_model.onnx in workspace_path (default is nc_workspace)" \
                "for newly generated node name")
            for idx, node in enumerate(model.graph.node):
                if node_names.count(node.name) > 1:
                    node.name = node.op_type + '_nc_rename_' + str(idx)
            onnx.save(model, os.path.join(self.work_space, "renamed_model.onnx")) 
        return model
    
    def _revert_fusedconv(self, model):
        from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg
        from onnx import onnx_pb as onnx_proto
        new_nodes = []
        remove_nodes = []
        for node in model.model.graph.node:
            if node.op_type == 'FusedConv':
                kwargs = {}
                activation_params = None
                for attr in node.attribute:
                    if attr.name == 'activation':
                        activation_type = attr.s.decode('utf-8')
                    elif attr.name == 'activation_params':
                        continue
                    else:
                        kwargs.update(attribute_to_kwarg(attr))
                if activation_type in ['Relu', 'Clip']:
                    continue
                conv = onnx.helper.make_node(
                    'Conv', node.input, [node.name], node.name.split('fused ')[-1], **kwargs)
                activation_input = conv.output

                activation = onnx.helper.make_node(activation_type,
                    conv.output, node.output, '_'.join((conv.name, activation_type)))
                new_nodes.extend([conv, activation])
                remove_nodes.append(node)
        model.model.graph.node.extend(new_nodes)
        for node in remove_nodes:
            model.model.graph.node.remove(node)
        model.update()
        return model
    
def strategy_registry(cls):
    """Class decorator used to register all TuneStrategy subclasses.

    Args:
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """
    assert cls.__name__.endswith(
        'TuneStrategy'
    ), "The name of subclass of TuneStrategy should end with \'TuneStrategy\' substring."
    if cls.__name__[:-len('TuneStrategy')].lower() in STRATEGIES: # pragma: no cover
        raise ValueError('Cannot have two strategies with the same name')
    STRATEGIES[cls.__name__[:-len('TuneStrategy')].lower()] = cls
    return cls

@strategy_registry
class ConfigTuneStrategy(TuneStrategy):
    """The auto tuning strategy.

    There are three stages executed by auto strategy sequentially,
    and the tuning process ends once the condition meets the exit policy.
    """

    def __init__(self,
                 model,
                 conf,
):

        super().__init__(model=model,
                         conf=conf,
           )
        logger.info(f"*** Initialize auto tuning")
        self.strategies_sequence = ['conservative', 'basic']
        self.conf = conf

        self.config = self._initialize_config(conf) 
        self.model = model
        self._framework = None
 
        self._capability = None
        
    @property
    def capability(self):
        """Gets the capability."""
        return self._capability
    
    @property
    def framework(self):
        """Gets the framework."""
        return self._framework
    
    
    @framework.setter
    def framework(self, value):
        """Sets the framework.

        Args:
            value: The new value for the framework.
        """
        self._framework = value
    
    
    
    def check_q_func(self):
        return 
    def _set_objectives(self):
        return 
    
    @capability.setter
    def capability(self, value):
        """Sets the capability.

        Args:
            value: The new value for the capability.
        """
        self._capability = value
        
    def _initialize_config(self, conf):
        """Init the tuning config based on user conf.

        Args:
            conf: User config

        Returns:
            Tuning config
        """
        config = conf.quantization
        config.diagnosis = getattr(config, 'diagnosis', None)
        
        return config
        
    def build_tuning_space(self, config):
        """Create the tuning space.

        Create the tuning space based on the framework capability and user configuration.

        Args:
            config: The Conf class instance includes all user configurations.
        """
        calib_sampling_size_lst = self.config.calibration_sampling_size
        calib_sampling_size_lst = [int(calib_sampling_size) for calib_sampling_size in calib_sampling_size_lst]
        if self.calib_dataloader:
            self.calib_iter = [math.ceil(int(x) / self.calib_dataloader.batch_size) \
                               for x in calib_sampling_size_lst]
        else:
            self.calib_iter = 1
        # create tuning space
        adaptor_cap = {
            'calib': {'calib_sampling_size': calib_sampling_size_lst},
            'op': self.capability['opwise']
        }
        tuning_space = TuningSpace(adaptor_cap, conf=config, framework=self.framework)
        return tuning_space


    
    def initial_tuning_cfg(self):
        """Init the tuning config.

        Initialize the tuning config according to the quantization approach.

        Returns:
            op_item_dtype_dict (OrderedDict): key is (op_name, op_type); value is quantization mode.
            quant_mode_wise_items (OrderedDict): key is quant_mode/precision; value is item list.
            initial_op_tuning_cfg (OrderedDict): key is (op_name, op_type); value is the initialized tuning config.
        """
        from neural_compressor.strategy.utils.constant import auto_query_order, static_query_order, dynamic_query_order
        from neural_compressor.strategy.utils.tuning_space import initial_tuning_cfg_with_quant_mode
        if self.config.approach == 'post_training_auto_quant':
            query_order = auto_query_order
        elif self.config.approach == 'post_training_dynamic_quant':
            query_order = dynamic_query_order
        elif self.config.approach == 'post_training_static_quant':
            query_order = static_query_order
        elif self.config.approach == 'quant_aware_training':
            query_order = auto_query_order

        quant_mode_wise_items = OrderedDict() # mode, op_item_lst
        pre_items = set()
        # Collect op items supported the specified mode.
        for quant_mode in query_order:
            items = self.tuning_space.query_items_by_quant_mode(quant_mode)
            filtered_items = list(filter(lambda item: item not in pre_items, items))
            pre_items = pre_items.union(set(items))
            quant_mode_wise_items[quant_mode] = filtered_items

        def initial_op_quant_mode(items_lst, target_quant_mode, op_item_dtype_dict):
            for item in items_lst:
                op_item_dtype_dict[item.name] = target_quant_mode

        op_item_dtype_dict = OrderedDict()
        for quant_mode, quant_mode_items in quant_mode_wise_items.items():
            initial_op_quant_mode(quant_mode_items, quant_mode, op_item_dtype_dict)

        initial_op_tuning_cfg = {}
        for op_name_type, quant_mode in op_item_dtype_dict.items():
            initial_op_tuning_cfg[op_name_type] = initial_tuning_cfg_with_quant_mode(op_name_type,
                                                                                     quant_mode,
                                                                                     self.tuning_space)
        return op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg


    def next_tune_cfg(self):
        """Generate and yield the default tuning config.

        Returns:
            tune_config (dict): A dict containing the tuning configuration for quantization.
        """
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        _, _, op_tuning_cfg = self.initial_tuning_cfg()
        op_tuning_cfg['calib_sampling_size'] = calib_sampling_size_lst[0]
        logger.info(f"Quantize the model with default config.")
        yield op_tuning_cfg
        
        
    def _tune_cfg_converter(self, op_tuning_cfg):
        """Convert op_tuning_cfg for adaptor.

        Args:
            op_tuning_cfg (Dict): the op tuning config.
        """
        tune_cfg = {'op': OrderedDict()}
        for op_name_type, op_config in op_tuning_cfg.items():
            if isinstance(op_config, OpTuningConfig):
                tune_cfg['op'][op_name_type] = op_config.get_state()
                op_cap_lst = self.capability['opwise'][op_name_type]
                # Add pattern for diagnosis
                for op_cap in op_cap_lst:
                    if 'pattern' in op_cap:
                        op_pattern = {}
                        op_pattern['sequence'] = op_cap['pattern']['sequence'][0] if\
                            'sequence' in op_cap['pattern'] else None
                        op_pattern['precision'] = op_cap['pattern']['precision'][0] if\
                            'precision' in op_cap['pattern'] else None
                        tune_cfg['op'][op_name_type]['pattern'] = op_pattern
            else:
                tune_cfg[op_name_type] = op_config
        tune_cfg['calib_sampling_size'] = op_tuning_cfg['calib_sampling_size']
        if self.calib_dataloader is not None:
            tune_cfg['calib_iteration'] =  math.ceil(int(tune_cfg['calib_sampling_size']) / \
                                                    self.calib_dataloader.batch_size)
        else:
            tune_cfg['calib_iteration'] = 1
        tune_cfg['approach'] = self.config.approach
        # Add the recipe config
        tune_cfg['recipe_cfgs'] = tune_cfg.get('recipe_cfgs', {})
        # For not tuning recipe, tune cfg use it directly
        tune_cfg['recipe_cfgs'].update(self._not_tuning_recipes_values)
        # WA for get the smooth quant args
        if 'smooth_quant_args' in self.config.recipes:
            tune_cfg['recipe_cfgs']['smooth_quant_args'] = self.config.recipes['smooth_quant_args']
        # For tuning recipe, use the default value if it not specified by recipe tuning sampler.
        for recipe_name, recipe_val in self._tuning_recipes_default_values.items():
            if recipe_name not in tune_cfg['recipe_cfgs']:
                tune_cfg['recipe_cfgs'][recipe_name] = recipe_val
        return tune_cfg
    
    def _initialize_algo_scheduler(self):
        algo_scheduler = AlgorithmScheduler(self.config.recipes)
    
        return algo_scheduler
    
    def _prepare_tuning(self):

        # """Prepare to tune and avoid repeated initialization of the adaptor and tuning space."""
        framework, framework_specific_info = self._set_framework_info()
        
        # self.adaptor = self.adaptor or FRAMEWORKS[framework](framework_specific_info)
        self.adaptor = EONNXRUNTIMEAdaptor(framework_specific_info)
        self.framework = self.framework or framework
       
        # self.cur_best_acc = self.cur_best_acc or self.initial_best_acc()
        # # query capability and build tuning space
        self.capability = self.capability or self.adaptor.query_fw_capability(self.model)
        # logger.debug(self.capability)

        self.tuning_space = self.tuning_space or self.build_tuning_space(self.config)
        self.algo_scheduler = self.algo_scheduler or self._initialize_algo_scheduler()
     
    def _set_framework_info(self, q_dataloader=None, q_func=None):
        framework_specific_info = {'device': getattr(self.config, 'device', None),
                                   'approach': getattr(self.config, 'approach', None),
                                   'random_seed': options.random_seed,
                                   'performance_only': self._not_tuning}
        framework = self.config.framework.lower()

        framework_specific_info.update({'backend': self.config.backend})
        framework_specific_info.update({'format': getattr(self.config, 'quant_format', None)})
        framework_specific_info.update({'domain': getattr(self.config, 'domain', None)})

        self.mixed_precision_mode = isinstance(self.config, MixedPrecisionConfig)

        if 'tensorflow' in framework:
            framework_specific_info.update(
                {"inputs": self.config.inputs,
                 "outputs": self.config.outputs,
                 'workspace_path': options.workspace,
                 'recipes': self.config.recipes,
                 'use_bf16': self.config.use_bf16 if self.config.use_bf16 is not None else False})
            for item in ['scale_propagation_max_pooling', 'scale_propagation_concat']:
                if framework_specific_info['recipes'] and item not in framework_specific_info['recipes']:
                    framework_specific_info['recipes'].update({item: True})
            if self.config.backend == 'itex':
                framework = 'tensorflow_itex'
        if 'keras' in framework:
            framework_specific_info.update({
                 'workspace_path': options.workspace, })
        if framework == 'mxnet':
            framework_specific_info.update({"q_dataloader": q_dataloader})
        if 'onnx' in framework.lower():
            if self.mixed_precision_mode:
                framework_specific_info.update({"approach": "post_training_dynamic_quant"})
            framework_specific_info.update({"deploy_path": os.path.dirname(self.deploy_path)})
            framework_specific_info.update({'workspace_path': options.workspace})
            framework_specific_info.update({'recipes': self.config.recipes})
            framework_specific_info.update({'reduce_range': self.config.reduce_range})
            framework_specific_info.update({'recipes': self.config.recipes})
            if framework_specific_info['backend'] in ['onnxrt_trt_ep', 'onnxrt_cuda_ep'] and \
                'gpu' not in framework_specific_info['device']:
                logger.warning('Please set device to gpu during using backend {}.'.format(self.config.backend))
                sys.exit(0)
            if framework.lower() == 'onnxrt_qdq' or \
                framework_specific_info['backend'] == 'onnxrt_trt_ep':
                framework_specific_info.update({'format': 'QDQ'})
                framework = 'onnxrt_qdq'
        if framework == 'pytorch_ipex' or framework == 'pytorch' or framework == 'pytorch_fx':
            if self.config.backend == 'ipex':
                framework = 'pytorch_ipex'
            elif self.config.backend == 'default':
                framework = 'pytorch_fx'
            if self.mixed_precision_mode:
                framework_specific_info.update({"approach": "post_training_dynamic_quant"})
            framework_specific_info.update({'recipes': self.config.recipes})
            framework_specific_info.update({"q_dataloader": q_dataloader})
            framework_specific_info.update({"use_bf16": self.config.use_bf16 \
                            if self.config.use_bf16 is not None else True})
            framework_specific_info.update(
                {"workspace_path": os.path.dirname(self.deploy_path)})
            if self.config.op_name_dict is not None \
               and 'default_qconfig' in self.config.op_name_dict:
                framework_specific_info.update(
                    {"default_qconfig": self.config.op_name_dict['default_qconfig']})
            framework_specific_info.update({"q_func": q_func})
            framework_specific_info.update({"example_inputs": self.config.example_inputs})
        return framework, framework_specific_info
    
    def config_generate(self):
        
        self._prepare_tuning()   #initialization of the adaptor and tuning space   self.adaptor self.tuning_space

        for op_tuning_cfg in self.next_tune_cfg(): # √   
            
             tune_cfg = self._tune_cfg_converter(op_tuning_cfg)    # √
             q_config = self.adaptor.quantize_config_generate(copy.deepcopy(tune_cfg),self.model)
        return q_config
    
    
    
if __name__=='__main__':
    # qdqmodel = onnx.load("/home/wrq/quant_SDK/quant_uint8.onnx")
    qdqmodel = onnx.load("/home/wrq/neural-compressor/examples/onnxrt/image_recognition/resnet50_torchvision/quantization/ptq_static/resnet50_quant_int8_conv_acc78.onnx")
    user_config = PostTrainingQuantConfig(   
            quant_format="QOperator"
    )
    graph = qdqmodel.graph
    nodes = graph.node

    # bp()
    # for node in nodes:
    #     print("Node Name:", node.name)
    #     print("Node Op Type:", node.op_type)
    #     print("Node Inputs:", node.input)
    #     print("Node Outputs:", node.output)
    #     print("-" * 20)

    wrapped_model = Model(qdqmodel, conf=user_config)   
    print(wrapped_model.model.opset_import)
  
    # onnx.save(wrapped_model,"resnet50_quant_int8_acc78_INC.onnx")
    # print(onnx.helper.printable_graph(qdqmodel.graph))
    user_configs = _Config(quantization=user_config, benchmark=None, pruning=None, distillation=None, nas=None)
    # for node in wrapped_model.nodes():
    #     with open('test.txt','a') as file0:
    #          print(node.name,file=file0)



        # print(node.name)
        # print("#############")

    # onnx.save(wrapped_model,"resnet50_quant_int8_acc78_INC.onnx")
    
    config_strategy = ConfigTuneStrategy(wrapped_model, user_configs)

    # onnx.save(wrapped_model,"resnet50_quant_int8_acc78_INC.onnx")

#   print(wrapped_model.graph_info)
    
    q_config = config_strategy.config_generate()
    # opset=17

    convertor = Convertor(model= wrapped_model ,q_config=q_config)

    convertor.convert_model()
    #opset=17
    qlinearmodel = convertor.model.model
    #opset=17
    print(qlinearmodel.opset_import)
    onnx = LazyImport('onnx')
    onnx.save(qlinearmodel,"/home/wrq/neural-compressor/examples/onnxrt/image_recognition/resnet50_torchvision/quantization/ptq_static/INC_resnet50_quant_int8_conv_acc78.onnx")
    # print(qlinearmodel)
    print("successs")
    
   
    
    
    


