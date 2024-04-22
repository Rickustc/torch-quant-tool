


import onnx

model = onnx.load("/home/wrq/tvm_demo/evas_workspace/tmp_model.onnx")


for node in model.graph.node:
    if node.op_type == "Conv":
        import pdb
        pdb.set_trace()