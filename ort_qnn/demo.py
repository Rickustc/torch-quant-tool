import onnxruntime as ort
# Create a session with QNN EP using HTP (NPU) backend.
import numpy as np
import onnx
# sess_options = ort.SessionOptions()
# sess_options.optimized_model_filepath = "/home/wrq/vit_fx_ort_int8model_double.onnx"

# sess_options_config_entries = {}
# sess_options_config_entries["session.qdqisint8allowed"] = "1"    # trtqdq to int8qdq
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# for key, value in sess_options_config_entries.items():
#     sess_options.add_session_config_entry(key, value)

# model_path = "/home/wrq/vit_fx_ort_int8model.onnx"
# sess = ort.InferenceSession(model_path,
#                             sess_options,
#                             providers=['QNNExecutionProvider'], 
#                             provider_options=[{'backend_path':'QnnCpu.dll'}]
# )

model_path = "/home/wrq/ort_qnn/modified_quantized.onnx"
model_onnx = onnx.load(model_path)


for node in model_onnx.graph.node:
    if not node.domain :
        node.domain = "com.microsoft"
    print("op_type:{}, domain:{}".format(node.op_type,node.domain))

onnx.save()

onnx.checker.check_model(model_onnx)
sess = ort.InferenceSession(model_path)
input_nodes = sess.get_inputs()
output_nodes = sess.get_outputs()
for i in range(0, len(input_nodes)):
    print("[INFO] Model input name <{}>:".format(i), input_nodes[i].name, "input shape :",input_nodes[i].shape, input_nodes[i].type)
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