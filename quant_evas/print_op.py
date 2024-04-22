import onnx

model = onnx.load("/home/wrq/quant_evas/float.onnx")
op_list = []
for node in model.graph.node:
    if node.op_type not in op_list:
        op_list.append(node.op_type)
        
print(op_list)
        
        
