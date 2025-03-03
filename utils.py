import subprocess

def start_ffmpeg(video_stream_url, frame_width, frame_height, fps=10):
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_stream_url,
        "-loglevel", "quiet",
        "-an",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={frame_width}:{frame_height},fps={fps}",
        "-"
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**8)
    return process
