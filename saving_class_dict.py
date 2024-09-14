from ultralytics import YOLOv10
import cv2
import os
from collections import defaultdict
import json
from tqdm import tqdm  # Import tqdm for progress tracking

model = YOLOv10('pretrained/yolov10l.pt')
idx = 0
object_dict = defaultdict(lambda: defaultdict(lambda: {"count": 0}))

keyframes_dir = 'keyframes'

image_paths = []
for root, dirs, files in os.walk(keyframes_dir):
    for file in files:
        image_paths.append(os.path.join(root, file))

for img_path in tqdm(image_paths, desc="Processing Images"):
    img = cv2.imread(img_path)
    results = model.predict(img, conf=0.20)
    
    detected = False 

    for result in results:
        for box in result.boxes:
            object_dict[idx][result.names[int(box.cls[0])]]["count"] += 1
            detected = True  # Set detected flag to True
            # Uncomment the following lines if you need to store bounding boxes
            # object_dict[idx][result.names[int(box.cls[0])]]["bbox"].append(
            #     [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])]
            # )

    if not detected:
        object_dict[idx] = {}

    idx += 1

with open('dict.json', 'w') as f:
    json.dump(object_dict, f, indent=4)
