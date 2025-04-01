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

# Get input video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Speed measurement parameters
last_positions = {}  # Stores last known position and time for each vehicle
vehicle_speeds = {}  # Stores computed speed for each vehicle
speed_sums = {}  # Stores cumulative speed sum for averaging
speed_counts = {}  # Stores count of speed calculations per vehicle
saved_vehicles = set()  # Stores vehicles already saved as images

distance_per_pixel = 10 / 70  # Assumed conversion from pixels to meters

# Create output directories
output_folder = "../output/images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize CSV storage
speed_data = []

# Fixed resolution for saved vehicle images (e.g., 400x400)
fixed_resolution = (400, 400)

# Set up VideoWriter to save output video with the same resolution as input video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can choose another codec if needed
out_video_path = "../output/output.mp4"
out = cv2.VideoWriter(
    out_video_path, fourcc, 20.0, (frame_width, frame_height)
)  # Match the input frame size

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

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

        # Draw bounding box on video (no resolution changes in video)
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

        # Save image of unique vehicle with adjusted resolution if not already saved
        if obj_id not in saved_vehicles:
            # Crop the vehicle image from the frame
            vehicle_img = frame[y1:y2, x1:x2]

            # Resize the cropped vehicle image to a fixed resolution (e.g., 400x400)
            resized_vehicle_img = cv2.resize(vehicle_img, fixed_resolution)

            # Save the resized vehicle image
            vehicle_filename = f"{output_folder}/vehicle_{obj_id}.jpg"
            cv2.imwrite(vehicle_filename, resized_vehicle_img)
            saved_vehicles.add(obj_id)

    # Write the frame to the output video
    out.write(frame)

    # Show the frame with bounding boxes (no resolution changes in video)
    cv2.imshow("Speed Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Compute and save average speeds to CSV
for obj_id in speed_sums:
    avg_speed = speed_sums[obj_id] / speed_counts[obj_id]
    speed_data.append([obj_id, round(avg_speed, 2)])

df = pd.DataFrame(speed_data, columns=["Vehicle ID", "Average Speed (Km/h)"])
df.to_csv("../output/vehicle_speeds.csv", index=False)

print(
    f"Processing complete. Unique vehicle images with adjusted resolution and speed data saved.\nOutput video saved to {out_video_path}"
)
