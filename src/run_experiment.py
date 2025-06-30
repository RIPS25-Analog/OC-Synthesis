from ultralytics import YOLO

model = YOLO("yolo11n.pt", task='detect')
print(model.info()) 

results = model.train(data='new_data.yaml', epochs=100, freeze=23,)