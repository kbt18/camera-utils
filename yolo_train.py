from ultralytics import YOLO

# Load a model
model = YOLO("yolov6n.yaml")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="drones.yaml", epochs=100, imgsz=640)