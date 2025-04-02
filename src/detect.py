import cv2
import os
import pandas as pd
import time
from ultralytics import YOLO
from tracker import Tracker

model = YOLO("../models/yolov8s.pt")

cap = cv2.VideoCapture("../resources/test.mp4")

class_list = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

count = 0
tracker = Tracker()
down = {}
up = {}
counter_down = []
counter_up = []

red_line_y = 198
blue_line_y = 268
offset = 6

if not os.path.exists("output"):
    os.makedirs("output")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("../output/final_output.mp4", fourcc, 20.0, (1020, 500))

speed_data = {}  # Dictionary to store speed data

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data.cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    bbox_list = []

    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row)
        if class_list[class_id] in ["car", "motorcycle", "bus", "truck"]:
            bbox_list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(bbox_list)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[obj_id] = time.time()
        if (
            obj_id in down
            and blue_line_y < (cy + offset)
            and blue_line_y > (cy - offset)
        ):
            elapsed_time = time.time() - down.pop(obj_id)
            if obj_id not in counter_down:
                counter_down.append(obj_id)
                speed_kph = (10 / elapsed_time) * 3.6
                speed_data[obj_id] = int(
                    speed_kph
                )  # Store speed data in the dictionary
                cv2.putText(
                    frame,
                    f"{int(speed_kph)} Km/h",
                    (x4, y4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[obj_id] = time.time()
        if obj_id in up and red_line_y < (cy + offset) and red_line_y > (cy - offset):
            elapsed_time = time.time() - up.pop(obj_id)
            if obj_id not in counter_up:
                counter_up.append(obj_id)
                speed_kph = (10 / elapsed_time) * 3.6
                speed_data[obj_id] = int(
                    speed_kph
                )  # Store speed data in the dictionary
                cv2.putText(
                    frame,
                    f"{int(speed_kph)} Km/h",
                    (x4, y4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {obj_id}",
            (x3, y3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # If speed is already stored, display it constantly on the vehicle
        if obj_id in speed_data:
            cv2.putText(
                frame,
                f"{speed_data[obj_id]} Km/h",
                (x4, y4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

    cv2.rectangle(frame, (0, 0), (250, 90), (0, 255, 255), -1)
    cv2.line(frame, (172, 198), (774, 198), (0, 0, 255), 2)
    cv2.putText(
        frame, "Red Line", (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
    )
    cv2.line(frame, (8, 268), (927, 268), (255, 0, 0), 2)
    cv2.putText(
        frame, "Blue Line", (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
    )
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

    frame_filename = f"detected_frames/frame_{count}.jpg"
    cv2.imwrite(frame_filename, frame)
    out.write(frame)

    cv2.namedWindow("frames", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("frames", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
