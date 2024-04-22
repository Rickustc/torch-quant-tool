import onnx
import onnxsim




input_path = "/home/wrq/resnet_quant/resnet_quant_fx_reference_qdq.onnx"
# load and check
onnx_model = onnx.load(input_path)
model_sim, check_ok = onnxsim.simplify(onnx_model)
onnx.checker.check_model(model_sim)

for initializer in model_sim.graph.initializer:
    if initializer.data_type==2:
        initializer.data_type=3
        
onnx.save(model_sim,input_path)