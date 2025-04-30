from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run model
model.predict(source=0,conf=0.5,show=True)
