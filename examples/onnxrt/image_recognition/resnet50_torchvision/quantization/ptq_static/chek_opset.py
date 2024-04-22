import onnx

model = onnx.load("/home/wrq/neural-compressor/examples/onnxrt/image_recognition/resnet50_torchvision/quantization/ptq_static/resnet50_quant_int8_acc78_qlinear_fiery.onnx")



print(model.opset_import)