import cv2
import os

# defines the amount of frames skipped between saves
ratio = 110

def save_frames(video_path, folder_dest, ratio, start_index=226):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: it was not possible to open the video:\n{video_path}")
        return start_index

    cont = 0
    saved = 0
    current_index = start_index

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cont % ratio == 0:
            var_newName = os.path.join(folder_dest, f"img_{current_index}.png")
            cv2.imwrite(var_newName, frame)
            saved += 1
            current_index += 1

        cont += 1

    cap.release()
    print(f"Done: {saved} frames saved from '{video_path}' into '{folder_dest}'.")
    return current_index


# list with all videos to be framed
videos = [
    rf"C:MYPATH\V{i}.mp4"
    for i in range(1, 12) ## from portable 2
]

# destination folder
folder = r"C:MYPATH"

# to name the images
index_start = 672

for video in videos:
    index_start = save_frames(video, folder, ratio, start_index=index_start)
