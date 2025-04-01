import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("../models/yolov8n.pt")  # or yolov8n.onnx if using ONNX model

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30)  # Tracks vehicles for 30 frames if missing

# Define vehicle classes (common vehicle class IDs)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Camera calibration values (for example purposes)
distance_per_pixel = 0.05  # meters per pixel, obtained from calibration
fps = 30  # frames per second

# Speed limit (in meters per second) for overspeeding detection
SPEED_LIMIT = 60 / 3.6  # Convert 20 km/h to meters per second (approx 5.56 m/s)

# Open video
cap = cv2.VideoCapture("../resources/road.mp4")  # Replace with your video path

# Initialize CSV file to store speeds of all vehicles
output_dir = "../output/images"
os.makedirs(output_dir, exist_ok=True)

with open("../output/vehicle_speeds.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Track ID", "Speed (km/h)", "Timestamp", "Image Path"])  # Headers

# Variable to store previous position of the vehicle (for speed calculation)
previous_position = {}

# Output video setup
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 output
out = cv2.VideoWriter(
    "../output/output.mp4", fourcc, fps, (int(cap.get(3)), int(cap.get(4)))
)

# To keep track of image naming (unique IDs)
image_counter = 0
saved_vehicle_ids = set()  # Track vehicles that have been saved

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends

    # Perform detection using YOLO
    results = model(frame)
    detections = []

    # Collect valid detections (vehicles only)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            cls = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence score

            # Filter for vehicle classes and confidence threshold
            if cls in VEHICLE_CLASSES and conf > 0.3:
                detections.append(([x1, y1, x2, y2], conf, cls))  # Add detection

    # Update DeepSORT tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue  # Ignore unconfirmed tracks

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())  # Get tracked bounding box

        # Get current vehicle's position center (in pixels)
        current_position = (x1 + x2) // 2, (y1 + y2) // 2

        # Calculate the speed of the vehicle (if track_id exists in previous_position)
        speed_kmph = None  # Ensure that the speed_kmph is defined
        if track_id in previous_position:
            # Calculate the displacement in pixels
            displacement_pixels = np.linalg.norm(
                np.array(current_position) - np.array(previous_position[track_id])
            )

            # Convert displacement to real-world distance
            distance_traveled = displacement_pixels * distance_per_pixel  # in meters

            # Calculate speed in m/s
            speed_mps = distance_traveled * fps  # meters per second

            # Convert speed to km/h
            speed_kmph = speed_mps * 3.6  # kilometers per hour

            # Display speed on the frame
            cv2.putText(
                frame,
                f"Speed: {speed_kmph:.2f} km/h",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Check for overspeeding (speed limit threshold)
            if speed_mps > SPEED_LIMIT:
                # Display alert for overspeeding vehicle
                cv2.putText(
                    frame,
                    "OVERSPEEDING",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                # Optional: Add a sound alert or log message
                print(
                    f"Vehicle {track_id} is overspeeding! Speed: {speed_kmph:.2f} km/h"
                )

        # Save the vehicle image only once (if not already saved)
        if track_id not in saved_vehicle_ids:
            vehicle_image = frame[y1:y2, x1:x2]  # Crop the vehicle image from the frame
            image_name = f"{track_id}_{image_counter}.jpg"  # Use track ID and counter for unique naming
            image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(image_path, vehicle_image)  # Save image

            # Log speed data and image path to CSV file only if speed_kmph was calculated
            if speed_kmph is not None:
                with open("../output/vehicle_speeds.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            track_id,
                            f"{speed_kmph:.2f}",
                            cap.get(cv2.CAP_PROP_POS_FRAMES),
                            image_path,
                        ]
                    )

            # Increment image counter to ensure unique names
            image_counter += 1
            saved_vehicle_ids.add(track_id)  # Mark this vehicle as saved

        # Update the previous position of the vehicle
        previous_position[track_id] = current_position

        # Draw bounding box and track ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Write the frame with detections to the output video file
    out.write(frame)

    # Show the frame
    cv2.imshow("Vehicle Speed Monitoring", frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
