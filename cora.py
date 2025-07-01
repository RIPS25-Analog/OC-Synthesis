from ultralytics import YOLO

#model = YOLO("yolo11n.pt")
#model.train(data="env/lib/python3.10/site-packages/ultralytics/cfg/datasets/coco8.yaml", epochs=100, imgsz=640)

from ultralytics import SAM
# Load a model
model = SAM("sam2.1_b.pt")

#results = model("/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/000000000034.jpg", bboxes=[100, 100, 200, 200])
#results[0].save("result.jpg")

#trial = model("https://ultralytics.com/images/bus.jpg")
#trial[0].save("trial.jpg")

#trial = model("/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/IMG_8156.JPG")
#trial[0].save("mug_segment.jpg")

trial = model("/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/images/bhg-types-of-screwdrivers-hero_BRrOcNPR4EC9cfTcXJcJyv-42936dda0b424c2ca66a7858f56ef29b.jpg")
trial[0].save("screwdriver.jpg")
for tr in trial:
     tr.save("annotated_image.jpg")