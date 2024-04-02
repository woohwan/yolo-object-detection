from fastapi import FastAPI
import cv2
import json
import numpy as np
from ultralytics import YOLO

app = FastAPI()

@app.get("/")
async def read_root():
  return {"message": "Hello, World!"}

from fastapi import File, UploadFile
model = YOLO("best1.pt")

@app.post("/detect/")
async def detect_objects(file: UploadFile):
  # Process the uploaded image for object detection
  image_bytes = await file.read()
  image = np.frombuffer(image_bytes, dtype=np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  
  # Perform object detection with YOLOv8
  results = model.predict(image, conf=0.01)
  print("Executing output_fn from inference.py ...")
  infer = {}
  
  for r in results:
    # print(r.boxes)
    if r.boxes is not None:
        infer['boxes'] = r.boxes.numpy().data.tolist()
    if r.masks is not None:
        infer['masks'] = r.masks.numpy().data.tolist()
    if r.keypoints is not None:
        infer['keypoints'] = r.keypoints.numpy().data.tolist()
    if r.probs is not None:
        infer['probs'] = r.probs.numpy().data.tolist()

  return json.dumps(infer)