######
# a 18secs video had ~1100 frames

import cv2
import os

##defines the amount of frames by video
ratio = 24

def save_frames(video_path, folder_dest, ratio):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error : it was not possible open the video:\n{video_path}")
        return
    cont = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cont % ratio == 0:
            var_newName = os.path.join(folder_dest, f"img_{(cont//ratio)}.png")
            cv2.imwrite(var_newName, frame)
            saved += 1
        cont += 1
    cap.release()
    print(f"Done: {saved} frames saved on '{folder_dest}'.")

# EXEMPLO DE USO
video = r"PUT_THE_VIDEO_PATH"
folder = r"PUT_THE_FOLDER_PATH"

save_frames(video, folder, ratio)
