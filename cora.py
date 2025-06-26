from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="env/lib/python3.10/site-packages/ultralytics/cfg/datasets/coco8.yaml", epochs=100, imgsz=640)
