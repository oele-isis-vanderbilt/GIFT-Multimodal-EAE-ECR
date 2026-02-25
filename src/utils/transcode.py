import cv2
import os
from .video import get_video_framerate

def transcode(video_path):
    folder_path = os.path.join(os.path.dirname(video_path), "ENCODED")
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    filename = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(folder_path, filename+"_ENCODED.avi")

    fps = get_video_framerate(video_path)
    vs = cv2.VideoCapture(video_path)
    writer = None

    while True:
        ret, frame = vs.read()
        if not ret or frame is None:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
        writer.write(frame)

    vs.release()
    writer.release()
    return out_path
