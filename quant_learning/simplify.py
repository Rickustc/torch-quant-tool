import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load("/home/wrq/neural-compressor/examples/onnxrt/image_recognition/resnet50_torchvision/quantization/ptq_static/resnet50_quant_int8_conv_acc78.onnx")

# convert model
model_simp, check = simplify(model)
onnx.save(model_simp,"sim.onnx")
assert check, "Simplified ONNX model could not be validated"