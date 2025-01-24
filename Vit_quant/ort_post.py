import tvm.relay as relay
import onnx
import tvm
import onnxruntime as ort
import numpy as np





def _create_session_options(path):
 
    # target_platform == "arm"
    session_options_config_entries = {}
    session_options_config_entries["session.qdqisint8allowed"] = "1"

    session_options = ort.SessionOptions()
    # for key, value in session_options_config_entries.items():
    #     session_options.add_session_config_entry(key, value)

    session_options.graph_optimization_level = (
        # ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        #   ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
    session_options.optimized_model_filepath = path

    return session_options

# output model
output_path = "/home/wrq/resnet_quant/ort_resnet_quant_fx_reference_qdq_shift_scale.onnx"
# /home/wrq/resnet_quant/ort_resnet_quant_fx_reference_qdq.onnx
# path = "/home/wrq/resnet_quant/resnet_quant_fx_reference_qdq_shift_scale.onnx"
# set parma for InferenceSession
session_options  =_create_session_options(output_path)
backend = "CPUExecutionProvider"
sess = ort.InferenceSession("/home/wrq/resnet_quant/resnet_quant_fx_reference_qdq_shift_scale.onnx",
                            session_options,
                            providers=[backend],
                            )


# load and check
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)

for initializer in onnx_model.graph.initializer:
    if initializer.data_type==2:
        initializer.data_type=3
        
onnx.save(onnx_model,output_path)