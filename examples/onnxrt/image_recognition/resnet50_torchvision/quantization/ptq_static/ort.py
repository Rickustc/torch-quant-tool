
import onnxruntime as ort
import numpy as np
import onnx
from onnx.backend.test.case.node import expect
import warnings
warnings.filterwarnings("ignore")
from spox import argument, build, Tensor


session_options = ort.SessionOptions()
session_options.graph_optimization_level = (
# ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

session_options_config_entries = {}
session_options_config_entries["session.qdqisint8allowed"] = "1"
for key, value in session_options_config_entries.items():
    session_options.add_session_config_entry(key, value)
    
session_options.optimized_model_filepath = "/home/wrq/neural-compressor/examples/onnxrt/image_recognition/resnet50_torchvision/quantization/ptq_static/ORT_resnet50_quant_int8_conv_acc78.onnx"

ort.InferenceSession(
"/home/wrq/neural-compressor/examples/onnxrt/image_recognition/resnet50_torchvision/quantization/ptq_static/resnet50_quant_int8_conv_acc78.onnx",
providers=['CPUExecutionProvider'], sess_options=session_options,
)


def onnx_infer(onnx_model_path):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
    sess = ort.InferenceSession(onnx_model_path,providers=['CPUExecutionProvider'], sess_options=session_options)

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
    
    
# onnx_infer("/home/wrq/temp_model.onnx")

# node = onnx.helper.make_node(
#     "QLinearConv",
#     inputs=[
#         "x",
#         "x_scale",
#         "x_zero_point",
#         "w",
#         "w_scale",
#         "w_zero_point",
#         "y_scale",
#         "y_zero_point",
#     ],
#     outputs=["y"],
# )

# x = np.array(
#     [
#         [255, 174, 162, 25, 203, 168, 58],
#         [15, 59, 237, 95, 129, 0, 64],
#         [56, 242, 153, 221, 168, 12, 166],
#         [232, 178, 186, 195, 237, 162, 237],
#         [188, 39, 124, 77, 80, 102, 43],
#         [127, 230, 21, 83, 41, 40, 134],
#         [255, 154, 92, 141, 42, 148, 247],
#     ],
#     dtype=np.int8,
# ).reshape((1, 1, 7, 7))

# x_scale = np.float32(0.00369204697)
# x_zero_point = np.int8(132)

# w = np.array([0], dtype=np.int8).reshape((1, 1, 1, 1))

# w_scale = np.array([0.00172794575], dtype=np.float32)
# w_zero_point = np.array([255], dtype=np.int8)

# y_scale = np.float32(0.00162681262)
# y_zero_point = np.int8(123)

# output = np.array(
#     [
#         [0, 81, 93, 230, 52, 87, 197],
#         [240, 196, 18, 160, 126, 255, 191],
#         [199, 13, 102, 34, 87, 243, 89],
#         [23, 77, 69, 60, 18, 93, 18],
#         [67, 216, 131, 178, 175, 153, 212],
#         [128, 25, 234, 172, 214, 215, 121],
#         [0, 101, 163, 114, 213, 107, 8],
#     ],
#     dtype=np.int8,
# ).reshape((1, 1, 7, 7))

# expect(
#     node,
#     inputs=[
#         x,
#         x_scale,
#         x_zero_point,
#         w,
#         w_scale,
#         w_zero_point,
#         y_scale,
#         y_zero_point,
#     ],
#     outputs=[output],
#     name="test_qlinearconv",
# )
# onnx.save(node,"qlinearConv")