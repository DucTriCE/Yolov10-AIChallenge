from ultralytics import YOLOv10
import cv2
import os
from collections import defaultdict
import json

model = YOLOv10('pretrained/yolov10l.onnx')
idx = 0
object_dict = defaultdict(lambda: defaultdict(lambda: {"count": 0}))

keyframes_dir = 'keyframes'

for root, dirs, files in os.walk(keyframes_dir):
    for file in files:
        img_path = os.path.join(root, file)

        img = cv2.imread(img_path)
        results = model.predict(img, conf=0.05)
        
        for result in results:
            for box in result.boxes:
                object_dict[idx][result.names[int(box.cls[0])]]["count"] += 1
                # object_dict[idx][result.names[int(box.cls[0])]]["bbox"].append(
                #     [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])]
                # )
        idx += 1

with open('dict.json', 'w') as f:
    json.dump(object_dict, f, indent=4)
