import onnxruntime as ort
# import onnxruntime-gpu as ort
import numpy as np
import onnx
from onnx.backend.test.case.node import expect
# def qdq2qlinear(original_path,qlinear_path):
#     r"""    
#     convert qdqmodel to qlinearmodel
#     Args: 
#       * `original_path `qdqmodel path`
#       * `qlinear_path `path of qlinear model ` 
#       Before: q-dp-conv\
#               dp-weight
#       After:  Qconv 
#     """
    
#     session_options = ort.SessionOptions()
#     # to do how to understand graph_optimization_level
#     # https://github.com/microsoft/onnxruntime/blob/21ae86e4051751741ad9b92512595896853721b5/onnxruntime/core/optimizer/qdq_transformer/qdq_s8_to_u8.cc#L62
#     # https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions
#     session_options.graph_optimization_level = (
#         ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#         #   ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
#         )
    
#     # to make qdq allow int8: 
#     # export ONNXRUNTIME_SESSION_OPTIONS_qdq_is_int8_allowed=1
#     config_key = "session.qdqisint8allowed"
#     config_value = "1"  # You can set the value according to your needs

#     session_options.add_session_config_entry(config_key, config_value)
   
#     session_options.optimized_model_filepath = qlinear_path
#     # session_options.QDQIsInt8Allowed = True
#     sess = ort.InferenceSession(
#     original_path,
#     providers=['CPUExecutionProvider'], sess_options=session_options,

#     )
    
def onnx_infer(onnx_model_path):
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
    print(onnx_result[0].shape)
    
# model = onnx.load("/home/wrq/0904/test.onnx")
# onnx.checker.check_model(model)
onnx_infer("/home/wrq/q_model.onnx")

# from onnx import version_converter, helper

# # A full list of supported adapters can be found here:
# # https://github.com/onnx/onnx/blob/main/onnx/version_converter.py#L21
# # Apply the version conversion on the original model
# converted_model = version_converter.convert_version(original_model,13)
# onnx.save(converted_model,"new_model_version12.onnx")
# onnx_infer("/home/wrq/neural-compressor/examples/onnxrt/image_recognition/resnet50_torchvision/quantization/ptq_static/_INC_resnet50_quant_int8_conv_acc78.onnx")
# from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

# # 开始量化
# quantize_static(input_model_path,
#                 output_model_path,
#                 dr,
#                 quant_format=QuantFormat.QDQ,
#                 per_channel=False,
#                 weight_type=QuantType.QInt8)
    
# qdq2qlinear("/home/wrq/quant_SDK/export/resnet50_quant_int8_linear.onnx","./export/qlinear_resnet50.onnx")
# onnx_infer("/home/wrq/neural-compressor/examples/onnxrt/image_recognition/resnet50_torchvision/quantization/ptq_static/_INC_resnet50_quant_int8_conv_acc78.onnx")
    


    


