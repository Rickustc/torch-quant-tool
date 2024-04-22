

import onnx


onnx_model = onnx.load("/home/wrq/quant_SDK/gkt_nuscenes_vehicle_kernel_7x1.onnx")
op_dict = {}
for node in onnx_model.graph.node:
    
    if node.op_type in op_dict:
        op_dict[node.op_type] += 1 
    else:
        op_dict[node.op_type] = 1
        
        
for key,value in op_dict.items():
    print(key,value)
    
    
    