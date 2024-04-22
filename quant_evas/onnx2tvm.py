import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
import onnxruntime


model_path = '/home/wrq/vit_after_opt.onnx'
sess = onnxruntime.InferenceSession(model_path)

# 获取输入名称和形状
input_details = sess.get_inputs()
for input_detail in input_details:
    print(f"Input Name: {input_detail.name}, Input Shape: {input_detail.shape}")
    

onnx_model = onnx.load(model_path)
input_name = "1"
shape_dict = {input_detail.name: input_detail.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)