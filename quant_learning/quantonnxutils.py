import onnxruntime as ort
import onnxruntime as ort
import numpy as np
import onnx
from onnx.backend.test.case.node import expect

def qdq2qlinear(original_path,qlinear_path):
    r"""    
    convert qdqmodel to qlinearmodel
    Args: 
      * `original_path `qdqmodel path`
      * `qlinear_path `path of qlinear model ` 
      Before: q-dp-conv\
              dp-weight
      After:  Qconv 
    """
    session_options = ort.SessionOptions()
    # to do how to understand graph_optimization_level
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
    session_options.optimized_model_filepath = qlinear_path

    sess = ort.InferenceSession(
    original_path,
    providers=['CPUExecutionProvider'], sess_options=session_options,
    )
    
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
    

    
    


    


