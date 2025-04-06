import cv2
import os
import math
import pandas as pd
from tracker import Tracker

# Paths and setup
video_path = "../resources/test.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Constants
DIST_BETWEEN_LINES_M = 10
RED_LINE_Y = 198
BLUE_LINE_Y = 268
OFFSET = 6

# Output setup
os.makedirs("output", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("../output/final_output_bgsub.mp4", fourcc, 20.0, (1020, 500))

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40)

# Tracking & data
tracker = Tracker()
frame_id = 0
down_frame = {}
up_frame = {}
counter_down = []
counter_up = []
speed_data = {}

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    frame = cv2.resize(frame, (1020, 500))
    roi = frame[100:400, 200:900]
    fgmask = fgbg.apply(roi)

    # Morphological operations to reduce noise
    _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Contour detection
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 900:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x + 200, y + 100, w, h])

    # Track vehicles
    boxes_ids = tracker.update(detections)

    for box in boxes_ids:
        x, y, w, h, obj_id = box
        cx, cy = x + w // 2, y + h // 2

        # Speed logic: downward (red -> blue)
        if RED_LINE_Y - OFFSET < cy < RED_LINE_Y + OFFSET:
            down_frame[obj_id] = frame_id

        if obj_id in down_frame and BLUE_LINE_Y - OFFSET < cy < BLUE_LINE_Y + OFFSET:
            frame_diff = frame_id - down_frame[obj_id]
            if obj_id not in counter_down:
                counter_down.append(obj_id)
                elapsed_time = frame_diff / fps
                speed_kph = (DIST_BETWEEN_LINES_M / elapsed_time) * 3.6
                speed_data[obj_id] = int(speed_kph)

        # Speed logic: upward (blue -> red)
        if BLUE_LINE_Y - OFFSET < cy < BLUE_LINE_Y + OFFSET:
            up_frame[obj_id] = frame_id

        if obj_id in up_frame and RED_LINE_Y - OFFSET < cy < RED_LINE_Y + OFFSET:
            frame_diff = up_frame[obj_id] - frame_id
            if obj_id not in counter_up:
                counter_up.append(obj_id)
                elapsed_time = abs(frame_diff) / fps
                speed_kph = (DIST_BETWEEN_LINES_M / elapsed_time) * 3.6
                speed_data[obj_id] = int(speed_kph)

        # Draw bounding box and info
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{obj_id}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        if obj_id in speed_data:
            cv2.putText(
                frame,
                f"{speed_data[obj_id]} Km/h",
                (x + w, y + h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    # Draw info lines and box
    cv2.rectangle(frame, (0, 0), (250, 90), (0, 255, 255), -1)
    cv2.line(frame, (172, RED_LINE_Y), (774, RED_LINE_Y), (0, 0, 255), 2)
    cv2.putText(
        frame,
        "Red Line",
        (172, RED_LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
    cv2.line(frame, (8, BLUE_LINE_Y), (927, BLUE_LINE_Y), (255, 0, 0), 2)
    cv2.putText(
        frame,
        "Blue Line",
        (8, BLUE_LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        frame,
        f"Down: {len(counter_down)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        frame,
        f"Up: {len(counter_up)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )

    # Save to video
    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Save results
df = pd.DataFrame(list(speed_data.items()), columns=["Vehicle_ID", "Speed_kmph"])
df.to_csv("../output/vehicle_speeds_final.csv", index=False)
