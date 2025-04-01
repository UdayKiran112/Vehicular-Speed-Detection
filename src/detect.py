import cv2
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

# Load YOLO model
model = YOLO("../models/yolov8s.pt")

# Vehicle classes to track
class_list = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# Initialize tracker
tracker = Tracker()

# Video input
cap = cv2.VideoCapture("../resources/road.mp4")

# Speed measurement parameters
last_positions = {}  # Stores last known position and time for each vehicle
vehicle_speeds = {}  # Stores computed speed for each vehicle
speed_sums = {}  # Stores cumulative speed sum for averaging
speed_counts = {}  # Stores count of speed calculations per vehicle
saved_vehicles = set()  # Stores vehicles already saved as images

distance_per_pixel = 10 / 70  # Assumed conversion from pixels to meters

# Create output directories
output_folder = "../output/vehicles"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize CSV storage
speed_data = []

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (1020, 500))

    # YOLO prediction
    results = model.predict(frame)
    detections = results[0].boxes.data.cpu().numpy()

    bbox_list = []
    for det in detections:
        x1, y1, x2, y2, _, class_id = det.astype(int)
        class_id = int(class_id)

        if 0 <= class_id < len(class_list) and class_list[class_id] in [
            "car",
            "motorcycle",
            "bus",
            "truck",
        ]:
            bbox_list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(bbox_list)

    for bbox in bbox_id:
        x1, y1, x2, y2, obj_id = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Speed calculation
        current_time = time.time()

        if obj_id in last_positions:
            prev_cx, prev_cy, prev_time = last_positions[obj_id]
            time_diff = current_time - prev_time

            if time_diff > 0:  # Avoid division by zero
                distance_pixels = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                distance_meters = distance_pixels * distance_per_pixel
                speed_kph = (distance_meters / time_diff) * 3.6  # Convert m/s to km/h

                # Store speed for averaging
                vehicle_speeds[obj_id] = int(speed_kph)
                if obj_id not in speed_sums:
                    speed_sums[obj_id] = 0
                    speed_counts[obj_id] = 0
                speed_sums[obj_id] += speed_kph
                speed_counts[obj_id] += 1

        last_positions[obj_id] = (cx, cy, current_time)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display vehicle ID and speed
        text = f"ID: {obj_id}"
        if obj_id in vehicle_speeds:
            text += f" | {vehicle_speeds[obj_id]} Km/h"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Save image of unique vehicle if not already saved
        if obj_id not in saved_vehicles:
            vehicle_img = frame[y1:y2, x1:x2]
            vehicle_filename = f"{output_folder}/vehicle_{obj_id}.jpg"
            cv2.imwrite(vehicle_filename, vehicle_img)
            saved_vehicles.add(obj_id)

    cv2.imshow("Speed Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Compute and save average speeds to CSV
for obj_id in speed_sums:
    avg_speed = speed_sums[obj_id] / speed_counts[obj_id]
    speed_data.append([obj_id, round(avg_speed, 2)])

df = pd.DataFrame(speed_data, columns=["Vehicle ID", "Average Speed (Km/h)"])
df.to_csv("../output/vehicle_speeds.csv", index=False)

print("Processing complete. Unique vehicle images and speed data saved.")
