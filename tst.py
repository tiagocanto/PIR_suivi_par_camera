from ultralytics import YOLO
import numpy as np
import cv2



model = YOLO("yolov8n_v3_50e.pt")
print(model.names)