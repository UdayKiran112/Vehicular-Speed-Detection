import cv2
import pandas as pd
import os
from tracker import Tracker

cap = cv2.VideoCapture("../resources/test.mp4")
count = 0
tracker = Tracker()
down_frame = {}
up_frame = {}
counter_down = []
counter_up = []
speed_data = {}

fps = cap.get(cv2.CAP_PROP_FPS)
DIST_BETWEEN_LINES_M = 10

RED_LINE_Y = 198
BLUE_LINE_Y = 268
OFFSET = 6

if not os.path.exists("output"):
    os.makedirs("output")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("../output/final_output_haar.mp4", fourcc, 20.0, (1020, 500))

# Load Haar cascades
car_cascade = cv2.CascadeClassifier("cars.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    frame = cv2.resize(frame, (1020, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = []

    # Detect using each cascade
    for cascade in [car_cascade]:
        vehicles = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in vehicles:
            detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)

    for box in boxes_ids:
        x, y, w, h, obj_id = box
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2

        if RED_LINE_Y - OFFSET < cy < RED_LINE_Y + OFFSET:
            down_frame[obj_id] = count

        if obj_id in down_frame and BLUE_LINE_Y - OFFSET < cy < BLUE_LINE_Y + OFFSET:
            frame_diff = count - down_frame[obj_id]
            if obj_id not in counter_down:
                counter_down.append(obj_id)
                elapsed_time = frame_diff / fps
                speed_kph = (DIST_BETWEEN_LINES_M / elapsed_time) * 3.6
                speed_data[obj_id] = int(speed_kph)

        if BLUE_LINE_Y - OFFSET < cy < BLUE_LINE_Y + OFFSET:
            up_frame[obj_id] = count

        if obj_id in up_frame and RED_LINE_Y - OFFSET < cy < RED_LINE_Y + OFFSET:
            frame_diff = up_frame[obj_id] - count
            if obj_id not in counter_up:
                counter_up.append(obj_id)
                elapsed_time = abs(frame_diff) / fps
                speed_kph = (DIST_BETWEEN_LINES_M / elapsed_time) * 3.6
                speed_data[obj_id] = int(speed_kph)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {obj_id}",
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
                0.8,
                (0, 255, 255),
                2,
            )

    # Draw lines
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

    # Vehicle counts
    cv2.putText(
        frame,
        f"Going Down - {len(counter_down)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        frame,
        f"Going Up - {len(counter_up)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    out.write(frame)
    cv2.imshow("frames", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

speed_df = pd.DataFrame(list(speed_data.items()), columns=["Vehicle_ID", "Speed_km_h"])
speed_df.to_csv("../output/vehicle_speeds_haar.csv", index=False)
