from ultralytics import YOLO

# Load a model
model = YOLO('/home/wrq/yolo/yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.val(data='coco.yaml', epochs=100, imgsz=640)