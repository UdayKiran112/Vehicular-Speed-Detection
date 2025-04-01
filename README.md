# Vehicle Speed Measurement Using Computer Vision

This project uses computer vision techniques to measure the speed of vehicles in a video using YOLOv8 for object detection and a custom tracker for vehicle tracking. It calculates the speed of each vehicle in km/h and saves the average speed to a CSV file. Additionally, each vehicle's image is captured and saved with adjusted resolution.

## Prerequisites

Make sure you have the following installed on your system:

- Python 3.x
- pip (Python package installer)

## Installation

Clone the repository and install the required dependencies by following these steps:

### 1. Clone the repository

```bash
git clone https://github.com/UdayKiran112/Vehicular-Speed-Detection.git

cd Vehicular-Speed-Detection
```

### 2. Create a virtual environment (optional but recommended)

```bash
python3 -m venv venv source venv/bin/activate # For Linux/MacOS
venv\Scripts\activate # For Windows
```

### 3. Install required Python packages

```bash
pip install -r requirements.txt
```

This will install all the required dependencies, including OpenCV, PyTorch, YOLOv8, and others.

## Files and Folders Structure

- `src/` — Source code for the project.
- `models/` — Contains the pre-trained YOLOv8 model (`yolov8s.pt`).
- `resources/` — Directory containing input videos (e.g., `road.mp4`).
- `output/` — Output folder where detected vehicle images, video with bounding boxes, and speed CSV are saved.
- `output/vehicles/` — Folder where individual vehicle images are saved.
- `output/vehicle_speeds.csv` — CSV file containing the average speeds of the detected vehicles.

## Running the Project

### 1. Prepare the input video

Place your input video (e.g., `road.mp4`) in the `resources/` folder. Ensure the video has the correct resolution and vehicle type.

### 2. Run the script

Once all dependencies are installed, run the Python script to start vehicle detection and speed measurement:
bash python src/main.py

The script will:

- Read the input video.
- Perform object detection on each frame using the YOLOv8 model.
- Track vehicles using a custom tracker.
- Calculate the speed of each vehicle based on movement between frames.
- Save images of each unique vehicle with adjusted resolution.
- Draw bounding boxes and vehicle IDs with speed on the frames.
- Save the processed frames into an output video.
- Save vehicle speeds to a CSV file (`output/vehicle_speeds.csv`).

### 3. Output

- The processed video with vehicle bounding boxes will be saved in `output/output_video.mp4`.
- Each detected vehicle's image with its adjusted resolution will be saved in `output/vehicles/`.
- The average speeds of vehicles are saved in `output/vehicle_speeds.csv`.

### Example output

- **Processed video**: `output/output_video.mp4`
- **Vehicle images**: Saved as `vehicle_<ID>.jpg` in the `output/vehicles/` folder.
- **Vehicle speed data**: Saved in `output/vehicle_speeds.csv` with columns: `Vehicle ID`, `Average Speed (Km/h)`.

## YOLO Model

The project uses YOLOv8, which is a state-of-the-art object detection model. The model file `yolov8s.pt` should be placed in the `models/` directory.

You can download the pre-trained YOLOv8 model from [Ultralytics YOLO GitHub](https://github.com/ultralytics/yolov8).

## Custom Tracker

The custom tracker (`Tracker` class) is responsible for tracking detected vehicles across frames. It uses the bounding box data to maintain consistent vehicle IDs across multiple frames.

## Speed Calculation

The vehicle speed is calculated based on the distance traveled in pixels between frames and the time difference between those frames. The distance in pixels is converted to meters using an assumed conversion factor (`10 meters / 70 pixels`), and the speed is calculated in km/h.

## Troubleshooting

- **Incorrect speed**: Ensure that the video resolution and the conversion factor (`distance_per_pixel`) are appropriate for your video.
- **Missing model**: If the YOLOv8 model file is not found, download it from the official YOLO repository.

### Acknowledgments

- [YOLOv8 Repository](https://github.com/ultralytics/yolov8)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
