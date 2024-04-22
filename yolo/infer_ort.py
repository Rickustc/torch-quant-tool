import onnxruntime as ort
import numpy as np
from PIL import Image
import numpy as np

random_input = np.random.rand(640, 3, 64, 64).astype('float32')
session = ort.InferenceSession('/home/wrq/yolov8.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: random_input})
detections = result[0]

print(detections.shape)

