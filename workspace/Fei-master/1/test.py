from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Display model information (optional)
model.info()

# Train the model with 2 GPUs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 2])

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model("path/to/bus.jpg")