import onnxruntime
import onnx
import numpy as np
from onnx.onnx_pb import TensorProto

model_path = '/home/wrq/quant_learning/quantized.onnx'

onnx_model = onnx.load(model_path)
for node in onnx_model.graph.node:
    if not node.domain :
        node.domain = "com.microsoft"
    # print("op_type:{}, domain:{}".format(node.op_type,node.domain))
    
# 遍历模型中的节点
# for node in onnx_model.graph.node:
#     print(node.op_type)
#     if node.op_type == 'QLinearAdd':
#         # 获取卷积层的权重和偏置的名称
#         for name in node.input:
#             if name.endswith("scale") or name.endswith("zero_point"):
#                 # 查找并修改scale和zp
#                 for initializer in onnx_model.graph.initializer:
                    
#                     if initializer.name == name:
#                         print("aaaaaaaa")
#                         import pdb
#                         pdb.set_trace()
#                         # # 修改type
#                         new_scale_zp  = initializer.float_data[:][0]
#                         initializer.float_data[:] = new_scale_zp
                     

# 保存修改后的模型
# onnx.save(onnx_model, 'modified_model.onnx')
# chehck ort
sess = onnxruntime.InferenceSession(model_path,
                                    providers=['QNNExecutionProvider'], 
                                    provider_options=[{'backend_path':'QnnHtp.dll'}])
# 获取输入名称和形状
input_details = sess.get_inputs()
for input_detail in input_details:
    print(f"Input Name: {input_detail.name}, Input Shape: {input_detail.shape}")
    
    
    

# model_path = "/home/wrq/ptq_int8_resnet.onnx"
# model = onnx.load(model_path)

# for initializer in model.graph.initializer:
#     print(initializer.name)
#     if initializer.data_type == 3: 
#         import pdb
#         pdb.set_trace()
