import cv2
import os
import numpy as np
from datetime import datetime
import yt_dlp

import roi
from tracker import CentroidTracker
from yolo_inference import load_model, run_inference
from utils import start_ffmpeg

ENABLE_TRACKING = False


# Setup snapshot folder
snapshot_folder = "snapshots"
os.makedirs(snapshot_folder, exist_ok=True)

# YouTube Live Stream URL
video_url = "https://www.youtube.com/live/GrEEoEmmrKs?si=ziheCq4_0J0akwr7"

# Extract direct video stream URL
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(video_url, download=False)
    video_stream_url = info["url"]

# Get stream resolution
cap = cv2.VideoCapture(video_stream_url)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Detected resolution: {frame_width}x{frame_height}")

# Initialize a default ROI if needed
roi.roi = [int(frame_width * 0.4), int(frame_height * 0.4),
       int(frame_width * 0.2), int(frame_height * 0.2)]

# Initialize tracker and timer dict
tracker = CentroidTracker(maxDisappeared=15)
objectTimers = {}

# Load YOLO model
model = load_model("yolov8x.pt")

# Start FFmpeg
ffmpeg_process = start_ffmpeg(video_stream_url, frame_width, frame_height, fps=10)

window_name = "Live Stream with ROI & Tracking"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, roi.roi_mouse_callback)

while True:
    raw_frame = ffmpeg_process.stdout.read(frame_width * frame_height * 3)
    if not raw_frame:
        break
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((frame_height, frame_width, 3)).copy()

    # Run YOLO inference
    results = run_inference(model, frame)
    rects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in ["car", "truck"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Check if box intersects ROI
                rx, ry, rw, rh = roi.roi
                if x1 < rx + rw and x2 > rx and y1 < ry + rh and y2 > ry:
                    rects.append((x1, y1, x2, y2))

    # Process detections for ROI and update tracker only if tracking is enabled
    if ENABLE_TRACKING:
        objects = tracker.update(rects)
        for objectID, centroid in objects.items():
            if objectID not in objectTimers:
                objectTimers[objectID] = datetime.now()
            elapsed = (datetime.now() - objectTimers[objectID]).total_seconds()
            text = f"ID {objectID}: {elapsed:.1f}s"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

        for objectID in list(objectTimers.keys()):
            if objectID not in objects:
                del objectTimers[objectID]

    # Draw ROI on frame
    rx, ry, rw, rh = roi.roi
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ffmpeg_process.terminate()
cv2.destroyAllWindows()
