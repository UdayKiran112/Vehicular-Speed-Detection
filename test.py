import torch
import cv2
import numpy as np
from ultralytics import YOLO

print(f"PyTorch Version: {torch.__version__}")
print(f"OpenCV Version: {cv2.__version__}")

model = YOLO("../models/yolov8n.pt")  # Check if YOLOv8 loads correctly

# Export the model to ONNX format
model.export(format="onnx")

# Load the exported ONNX model
print("YOLOv8 loaded successfully!")
