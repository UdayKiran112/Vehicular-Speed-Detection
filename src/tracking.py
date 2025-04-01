import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("../models/yolov8n.onnx")

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30)  # Tracks vehicles for 30 frames if missing

# Define vehicle classes
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Open video
cap = cv2.VideoCapture("../resources/traffic.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends

    # Perform detection
    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            cls = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence score

            if cls in VEHICLE_CLASSES and conf > 0.3:  # Threshold to filter weak detections
                detections.append(([x1, y1, x2, y2], conf, cls))  # Add detection

    # Pass detections to DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue  # Ignore unconfirmed tracks

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())  # Get tracked bounding box

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Vehicle Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
