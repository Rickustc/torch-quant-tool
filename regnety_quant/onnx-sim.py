
import onnx
import onnxsim
import onnxoptimizer
import onnx_tool


# new_model = onnxoptimizer.optimize(model)

input_path = "/home/wrq/regnety_quant/ort_regnet_quant_fx_reference_qdq_reference_shift_scale.onnx"
output_path = "/home/wrq/regnety_quant/onnx-sim.onnx"



onnx_tool.model_profile(input_path,saveshapesmodel='shapesssss.onnx')
# model_sim, check_ok = onnxsim.simplify(input_path,skipped_optimizers=True)
# onnx.checker.check_model(model_sim)

# # for initializer in model_sim.graph.initializer:
# #     if initializer.data_type==2:
# #         initializer.data_type=3
        
# onnx.save(model_sim,output_path)