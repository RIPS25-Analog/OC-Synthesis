from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data = "datasets/kikaben_data.yaml", 
    epochs = 100, 
    batch = 4, 
    freeze = 10
    )
print("\n\n\n\n\n")

results = model.val(data="datasets/kikaben_data.yaml")
print("\n\n\n\n\n")

# Print specific metrics
'''
print("Class indices with average precision:", results.ap_class_index)
print("Average precision for all classes:", results.box.all_ap)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean recall:", results.box.mr)'''

# Perform object detection on an image using the model
results = model("datasets/img.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")