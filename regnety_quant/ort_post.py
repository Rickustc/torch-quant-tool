import tvm.relay as relay
import onnx
import tvm
import onnxruntime as ort
import numpy as np
import onnxsim




def _create_session_options(path):
    """Args:
    path: optimized model output path
    """
    # target_platform == "arm"
    session_options_config_entries = {}
    session_options_config_entries["session.qdqisint8allowed"] = "1"
    # session_options_config_entries["session.x64quantprecision"]  ="0"  
    session_options = ort.SessionOptions()
    for key, value in session_options_config_entries.items():
        session_options.add_session_config_entry(key, value)
    
    session_options.graph_optimization_level = (
        # ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        #   ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
    session_options.optimized_model_filepath = path

    return session_options



input_path = "/home/wrq/regnety_quant/regnet_quant_fx_reference_qdq_reference.onnx"
# load and check
onnx_model = onnx.load(input_path)
model_sim, check_ok = onnxsim.simplify(onnx_model,skipped_optimizers=True )
onnx.checker.check_model(model_sim)

for initializer in model_sim.graph.initializer:
    if initializer.data_type==2:
        initializer.data_type=3
        
onnx.save(model_sim,input_path)



# output model
output_path = "/home/wrq/regnety_quant/ort_regnet_quant_fx_reference_qdq_reference_shift_scale.onnx"
# /home/wrq/resnet_quant/ort_resnet_quant_fx_reference_qdq.onnx
# path = "/home/wrq/resnet_quant/resnet_quant_fx_reference_qdq_shift_scale.onnx"
# set parma for InferenceSession
session_options  =_create_session_options(output_path)
backend = "CPUExecutionProvider"
sess = ort.InferenceSession(input_path,
                            session_options,
                            providers=[backend],
                            )



