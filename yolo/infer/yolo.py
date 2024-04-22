from ultralytics import YOLO
import torch
model = YOLO('yolov8s.yaml')

# checkpoint = torch.load("/home/fiery/wrq/yolov8s.pt")
# state_dict =  checkpoint["model"].state_dict()
# model.model.load_state_dict(state_dict, strict=False)

model = YOLO('yolov8s.pt')
# set opset=11
success = model.export(format='onnx',opset=11)