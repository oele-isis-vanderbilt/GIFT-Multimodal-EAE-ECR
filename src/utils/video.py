import cv2

def get_video_framerate(video):
    release = False
    if not isinstance(video, cv2.VideoCapture):
        video = cv2.VideoCapture(video)
        release = True
    fps = video.get(cv2.CAP_PROP_FPS)
    if release:
        video.release()
    
    if isinstance(fps, float) and fps > 0:
        return fps
    else:
        return None

def count_video_frames(video):
    release = False
    if not isinstance(video, cv2.VideoCapture):
        video = cv2.VideoCapture(video)
        release = True
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        total = 0
        while True:
            (grabbed, frame) = video.read()
            if not grabbed:
                break
            total += 1

    if release:
        video.release()
    else:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return total