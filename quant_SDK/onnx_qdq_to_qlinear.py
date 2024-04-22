
import copy
import os
from neural_compressor.adaptor.ox_utils.quantizer import Quantizer
import onnx
from neural_compressor.adaptor.ox_utils.operators import OPERATORS
from neural_compressor.model.onnx_model import ONNXModel
import onnxruntime as ort
import logging
logger = logging.getLogger("evas_qdqconvertor")
from importlib.util import find_spec
import sys
from onnxsim import simplify
import onnxoptimizer
import numpy as np
import warnings
from utils import _dump_model_op_stats

def check_model_in_ort(model_path):
    try:
        ort_infer_check(model_path)
    except Exception as e:
        print("exported model can not be run in onnxruntime")

def ort_infer_check(onnx_model_path):
    r"""    
    infer a qdqmodel using ort
    """

    sess = ort.InferenceSession(onnx_model_path)
    
    input_nodes = sess.get_inputs()
    output_nodes = sess.get_outputs()
    for i in range(0, len(input_nodes)):
        print("[INFO] Model input name <{}>:".format(i), input_nodes[i].name, "input shape :",
            input_nodes[i].shape, input_nodes[i].type)
    for i in range(0, len(output_nodes)):
        print("[INFO] Model output name <{}>:".format(i), output_nodes[i].name, 'output shape: ', output_nodes[i].shape)
    input_shape = (-1, int(sess.get_inputs()[0].shape[1]), int(sess.get_inputs()[0].shape[2]), int(sess.get_inputs()[0].shape[3]))

    dict_input_node={}
    list_input_node = []
    list_output_node=[]
    
    for i in range(0,len(input_nodes)): 
        # create fake input and save
        if('double' in input_nodes[i].type):
            img= np.random.randint(low=0, high=256, size=input_nodes[i].shape, dtype=np.uint8).astype(np.float64)
        else:
            img= np.random.randint(low=0, high=256, size=input_nodes[i].shape, dtype=np.uint8).astype(np.float32)
        dict_input_node[input_nodes[i].name]=img

        list_input_node.append(input_nodes[i].name)

    for i in range(0,len(output_nodes)):
        list_output_node.append(output_nodes[i].name)

    onnx_result=sess.run(list_output_node,dict_input_node)
    
    
    
class QDQQuantizer(Quantizer):
    """
    self.config: set: {op, dtype}
    self.model: a wrapped onnx model class which contian a onnx model
    
    """
    def __init__(self,model):
        # check is onnxmodel
        self.model = ONNXModel(model)
        self.pre_optimized_model = None
        self._input_name_to_nodes = {}
        self._output_name_to_node = {}
        
    def _rename_node(self, model):
        node_names = [i.name for i in model.graph.node]
        if len(set(node_names)) < len(node_names):
            logger.warning("This model has nodes with the same name, please check" \
                "renamed_model.onnx in workspace_path (default is nc_workspace)" \
                "for newly generated node name")
            for idx, node in enumerate(model.graph.node):
                if node_names.count(node.name) > 1:
                    node.name = node.op_type + '_nc_rename_' + str(idx)
            onnx.save(model, os.path.join(os.getcwd(), "renamed_model.onnx")) 
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
    # use onnx-simlify optimize
    def _onnxsim_pre_optimize(self,):
        model_opt,_= simplify(self.model.model)
       
        self.model.model = model_opt
    # use onnx-optimize optimize
    def _onnxopt_pre_optimize(self,):
        model_opt = onnxoptimizer.optimize(self.model.model)
        self.model.model = model_opt
     
        # self.model.topological_sort()
    
    # use ort optimize   
    def _ort_pre_optimize(self,  level=1):

        from neural_compressor import options
        from neural_compressor.adaptor.ox_utils.util import \
            remove_init_from_model_input, split_shared_bias
        
        remove_init_from_model_input(self.model)
        sess_options = ort.SessionOptions()
        #set optimize level
        optimization_levels = {
                'DISABLE_ALL': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                'ENABLE_BASIC': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                'ENABLE_EXTENDED': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                'ENABLE_ALL': ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                }
        level = 'ENABLE_BASIC'

        # sess_options.add_session_config_entry(config_key, config_value)
        sess_options.graph_optimization_level = optimization_levels[level]
        sess_options.optimized_model_filepath = "./evas_workspace/Optimized_model.onnx"
        if sys.version_info < (3,10) and find_spec('onnxruntime_extensions'): # pragma: no cover
            from onnxruntime_extensions import get_library_path
            sess_options.register_custom_ops_library(get_library_path())
        backend = "CPUExecutionProvider"
        ort.InferenceSession(self.model.model.SerializeToString(),
                                 sess_options,
                                 providers=[backend])


        tmp_model = onnx.load(sess_options.optimized_model_filepath)
        self.model.model_path = sess_options.optimized_model_filepath
        self.model.model = tmp_model
        self.model.model = self._rename_node(self.model.model)
        self.model = self._revert_fusedconv(self.model)
        self.model = split_shared_bias(self.model)
        # self.model.topological_sort()
        # check onnx graph
        onnx.checker.check_model(self.model.model)
        self.pre_optimized_model = self.model
        # from neural_compressor import options
        # from neural_compressor.adaptor.ox_utils.util import \
        #     remove_init_from_model_input, split_shared_bias
        
        # remove_init_from_model_input(self.model)
        # sess_options = ort.SessionOptions()
        # #set optimize level
        # optimization_levels = {
        #         'DISABLE_ALL': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        #         'ENABLE_BASIC': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        #         'ENABLE_EXTENDED': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        #         'ENABLE_ALL': ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        #         }
        # level = 'ENABLE_BASIC'

        # # sess_options.add_session_config_entry(config_key, config_value)
        # sess_options.graph_optimization_level = optimization_levels[level]
        # sess_options.optimized_model_filepath = "./evas_workspace/Optimized_model.onnx"
        # if sys.version_info < (3,10) and find_spec('onnxruntime_extensions'): # pragma: no cover
        #     from onnxruntime_extensions import get_library_path
        #     sess_options.register_custom_ops_library(get_library_path())
        # backend = "CPUExecutionProvider"

        # ort.InferenceSession(self.model.model.SerializeToString(),
        #                          sess_options,
        #                          providers=[backend])


        # tmp_model = onnx.load(sess_options.optimized_model_filepath)
        # self.model.model_path = sess_options.optimized_model_filepath
        # self.model.model = tmp_model
        # self.model.model = self._rename_node(self.model.model)
        # self.model = self._revert_fusedconv(self.model)
        # self.model = split_shared_bias(self.model)
        # # self.model.topological_sort()
        # # check onnx graph
        # onnx.checker.check_model(self.model.model)
        # self.pre_optimized_model = self.model
   
# convert qdqQuantizer to qlinear model
def onnx_qdq_to_qlinear(
    model,
    save_path
):
    """Export ONNX QDQ model into QLinearops model
    Args:
        model (ModelProto): fp32 qdq onnx model.
    """
    qdqQuantizer= QDQQuantizer(model) 
    # TODO: modify ort graph opt
    # graph opt using onnx simplifier
    qdqQuantizer._ort_pre_optimize()
    
    # set nodes as empty list      
    qdqQuantizer.new_nodes = []
    qdqQuantizer.remove_nodes = []
    qdqQuantizer.replace_input = []

    
    # convert ONNX QDQ model into QLinearops model
    for node in qdqQuantizer.model.nodes():
        if node.op_type in OPERATORS and node.op_type not in ['QuantizeLinear', 'DequantizeLinear']:
            # only support dynamic and static quant
            mode = "static"
            #node.name with quant(means ithas been int type) can be identify aiming to convert qdq to qlinear
            node.name = node.name + "_quant" 
            op_OPERATORS = OPERATORS[node.op_type](qdqQuantizer,node)
            op_OPERATORS.convert(mode)
        
    qdqQuantizer.model.graph().node.extend(qdqQuantizer.new_nodes)
    qdqQuantizer.model.remove_nodes(qdqQuantizer.remove_nodes)
    for node, old_input_name, new_input_name in qdqQuantizer.replace_input:
        qdqQuantizer.model.replace_node_input(node, old_input_name, new_input_name)
    qdqQuantizer.model.update()
    
    qdqQuantizer.model.topological_sort()
    # check onnx graph
    onnx.checker.check_model(qdqQuantizer.model.model)
    # set uint8 activation (y_zero_point = 0) to int8
    for initializer in qdqQuantizer.model.model.graph.initializer:
        if initializer.data_type==2:
            initializer.data_type=3
    

    # show quanted op info
    _dump_model_op_stats(qdqQuantizer.model.model)
            
    #save model
    onnx.save(qdqQuantizer.model.model,save_path)
    
    # check model in ort
    check_model_in_ort(save_path)
    
model_path = "/home/wrq/yolov8.onnx"
qdqQuantizer= QDQQuantizer(model_path) 

qdqQuantizer._ort_pre_optimize()

# set nodes as empty list      
qdqQuantizer.new_nodes = []
qdqQuantizer.remove_nodes = []
qdqQuantizer.replace_input = []
    
# convert ONNX QDQ model into QLinearops model
for node in qdqQuantizer.model.nodes():
    if node.op_type in OPERATORS and node.op_type not in ['QuantizeLinear', 'DequantizeLinear']:
        # only support dynamic and static quant
        mode = "static"
        import pdb
        pdb.set_trace()
        #node.name with quant(means ithas been int type) can be identify aiming to convert qdq to qlinear
        node.name = node.name + "_quant" 
        op_OPERATORS = OPERATORS[node.op_type](qdqQuantizer,node)
        op_OPERATORS.convert(mode)
        
qdqQuantizer.model.graph().node.extend(qdqQuantizer.new_nodes)
qdqQuantizer.model.remove_nodes(qdqQuantizer.remove_nodes)
for node, old_input_name, new_input_name in qdqQuantizer.replace_input:
    qdqQuantizer.model.replace_node_input(node, old_input_name, new_input_name)
qdqQuantizer.model.update()
# qdqQuantizer._onnxopt_pre_optimize()

# qdqQuantizer.model.topological_sort()
# check onnx graph
# onnx.checker.check_model(qdqQuantizer.model.model)
# # set uint8 activation (y_zero_point = 0) to int8
# for initializer in qdqQuantizer.model.model.graph.initializer:
#     if initializer.data_type==2:
#         initializer.data_type=3


# show quanted op info
# _dump_model_op_stats(qdqQuantizer.model.model)
# qdqQuantizer._onnxopt_pre_optimize()
save_path = "yolov8_int8.onnx"
#save model
onnx.save(qdqQuantizer.model.model,save_path)


    




        

    