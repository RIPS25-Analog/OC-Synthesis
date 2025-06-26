from ultralytics import YOLO

model = YOLO("yolo11n.pt")

#results = model.train(data="/home/szhang/RIPS25-AnalogDevices-ObjectDetection/my_data.yaml", epochs=3)
print("=======================================\n=======================================")
results = model.train(data="coco8.yaml", epochs=3)

#results = model.val(data="/home/szhang/RIPS25-AnalogDevices-ObjectDetection/my_data.yaml")
print("=======================================\n=======================================")
results = model.val(data="coco8.yaml")

# Print specific metrics
'''
print("Class indices with average precision:", results.ap_class_index)
print("Average precision for all classes:", results.box.all_ap)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean recall:", results.box.mr)'''

# Perform object detection on an image using the model
results = model("/home/szhang/RIPS25-AnalogDevices-ObjectDetection/img.jpg")

# Export the model to ONNX format
#success = model.export(format="onnx")