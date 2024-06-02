from ultralytics import YOLO
from roboflow import Roboflow
import cv2

# rf = Roboflow(api_key="********")
# project = rf.workspace("").project("")
# version = project.version(1)
# dataset = version.download("yolov8")
# -----------------------------------------------------------------
# model = YOLO("yolov8n.pt")

# results = model.train(data="MHT-1/data.yaml", epochs=300, imgsz=640)

model = YOLO("./runs/detect/train6/weights/best.pt")

# import matplotlib.pyplot as plt
# result = model.predict("./test.jpg")

# plt.figure(figsize=(12, 12))
# plt.imshow(cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB))
# plt.show()

video_paths = ["./car.mp4", "./dog.mp4", "./pigeon.mp4"]
caps = []

for path in video_paths:
    caps.append(cv2.VideoCapture(path))

for cap in caps:
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
