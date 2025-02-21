import cv2
import subprocess
import yt_dlp
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# Load YOLO model (best version)
model = YOLO("yolov8x.pt")  # ✅ Best accuracy model

# Define save folder for snapshots
snapshot_folder = "snapshots"
os.makedirs(snapshot_folder, exist_ok=True)

# YouTube Live Stream URL
video_url = "https://www.youtube.com/watch?v=62ua9iu8C-w"

# Extract the direct video stream URL
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(video_url, download=False)
    video_stream_url = info["url"]

# Get actual resolution from the stream
cap = cv2.VideoCapture(video_stream_url)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"✅ Detected Stream Resolution: {frame_width}x{frame_height}")

# Define the reference line (adjust based on video perspective)
line_y = int(frame_height * 0.6)  # 60% down the video height

# Dictionary to track objects that have already crossed the line
crossed_objects = {}

# Use FFmpeg to capture video at detected resolution
ffmpeg_cmd = [
    "ffmpeg",
    "-i", video_stream_url,
    "-loglevel", "quiet",
    "-an",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-vf", f"scale={frame_width}:{frame_height},fps=10",
    "-",
]

# Start FFmpeg process
ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**8)

while True:
    # Read raw frame data from FFmpeg output
    raw_frame = ffmpeg_process.stdout.read(frame_width * frame_height * 3)

    if not raw_frame:
        break

    # Convert raw frame to NumPy array (Make it writable to avoid OpenCV errors)
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((frame_height, frame_width, 3)).copy()

    # Run YOLO on the frame
    results = model(frame)

    # Draw the geographic reference line
    cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 0, 255), 2)  # Red line

    # Process detections
    new_crossing = False  # Flag for saving snapshot
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = f"{model.names[cls]} {conf:.2f}"  # Class name + confidence

            # Only detect cars, trucks, buses, motorcycles
            if model.names[cls] in ["car", "truck", "bus", "motorcycle"]:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check if object intersects the red line
                if y1 <= line_y <= y2:  # ✅ Object is touching the line
                    obj_id = f"{x1}_{x2}"  # Create a unique ID based on X position
                    if obj_id not in crossed_objects:  # If first time crossing
                        crossed_objects[obj_id] = True  # Mark as crossed
                        new_crossing = True  # Set flag to save snapshot

    # Save snapshot if a new car crosses the line
    if new_crossing:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(snapshot_folder, f"car_{timestamp}.jpg")
        cv2.imwrite(snapshot_path, frame)
        print(f"📸 Snapshot saved: {snapshot_path}")

    # Show the processed frame
    cv2.imshow("YOLOv8x Live Stream - Geographic Line", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit

ffmpeg_process.terminate()
cv2.destroyAllWindows()
