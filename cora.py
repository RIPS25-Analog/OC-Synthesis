from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)