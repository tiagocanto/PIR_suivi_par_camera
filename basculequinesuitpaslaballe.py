from ultralytics import YOLO
import numpy as np
import cv2
import threading
from queue import Queue, Empty

# --- CONFIG ---
VIDEO_FILES = ["cam1.mp4", "cam2.mp4", "cam3.mp4"]
RESIZE_WIDTH = 1280       # pour YOLO
FRAME_SKIP = 1            # traiter 1 frame sur 2
SMOOTH_WINDOW = 5         # nombre de frames pour lisser
SWITCH_THRESHOLD = 100    # seuil pour changer de caméra
MAX_DISPLAY_WIDTH = 1280
MAX_PLAYER_HEIGHT = 400   # ignore les joueurs trop proches

# --- YOLO ---
model = YOLO("yolov8n.pt")  # pré-entraîné

# --- FONCTIONS ---
def get_players_positions(frame, resize_width=RESIZE_WIDTH):
    h, w = frame.shape[:2]
    scale = resize_width / w
    frame_resized = cv2.resize(frame, (resize_width, int(h*scale)))
    results = model(frame_resized)[0]

    positions = []
    heights = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # personne
            x1, y1, x2, y2 = box.xyxy[0]
            height = y2 - y1
            if height / scale < MAX_PLAYER_HEIGHT:  # filtrer gros plan
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                positions.append((cx, cy))
    return positions

def compute_center(positions):
    if len(positions) == 0:
        return None
    x = int(np.mean([p[0] for p in positions]))
    y = int(np.mean([p[1] for p in positions]))
    return (x, y)

def compute_score(center, frame_shape):
    if center is None:
        return 0
    h, w, _ = frame_shape
    img_center = (w//2, h//2)
    dist = np.linalg.norm(np.array(center) - np.array(img_center))
    score = max(0, 500 - dist)
    return score

def show_frame(frame, window_name="Best Camera", max_width=MAX_DISPLAY_WIDTH):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    cv2.imshow(window_name, frame)

# --- THREAD DE LECTURE ---
def video_reader(filename, queue):
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            queue.put(None)
            break
        queue.put(frame)
    cap.release()

# --- INIT ---
queues = [Queue(maxsize=10) for _ in VIDEO_FILES]
threads = []
for i, file in enumerate(VIDEO_FILES):
    t = threading.Thread(target=video_reader, args=(file, queues[i]), daemon=True)
    t.start()
    threads.append(t)

last_scores = [ [0]*SMOOTH_WINDOW for _ in VIDEO_FILES ]
current_best = 0
frame_count = 0

# --- BOUCLE PRINCIPALE ---
while True:
    frame_count += 1
    frames = []
    scores = []

    # récupérer les frames
    for q in queues:
        try:
            frame = q.get(timeout=1.0)
        except Empty:
            frame = None
        frames.append(frame)

    # vérifier si toutes les vidéos sont finies
    if all(f is None for f in frames):
        break

    for i, frame in enumerate(frames):
        if frame is None:
            scores.append(last_scores[i][-1])
            continue

        # saut de frames pour performance
        if FRAME_SKIP > 0 and frame_count % (FRAME_SKIP + 1) != 0:
            scores.append(last_scores[i][-1])
            continue

        positions = get_players_positions(frame)
        center = compute_center(positions)
        score = compute_score(center, frame.shape)

        last_scores[i].append(score)
        if len(last_scores[i]) > SMOOTH_WINDOW:
            last_scores[i].pop(0)
        smooth_score = int(np.mean(last_scores[i]))
        scores.append(smooth_score)

        if center:
            cv2.circle(frame, center, 10, (0,0,255), -1)

    # Choisir caméra
    new_best = np.argmax(scores)
    if new_best != current_best:
        if scores[new_best] > scores[current_best] + SWITCH_THRESHOLD:
            current_best = new_best

    best_frame = frames[current_best]
    if best_frame is not None:
        cv2.putText(best_frame, f"Camera {current_best+1}",
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)
        show_frame(best_frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

# ------------------------------------- simulations avec images -------------------------------------
# charger images (simulent 3 frames de 3 caméras)
# frames = [
#     cv2.imread("cam1.jpg"),
#     cv2.imread("cam2.jpg"),
#     cv2.imread("cam3.jpg")
# ]

# def get_players_positions(frame):
#     results = modele(frame)[0]
#     positions = []

#     for box in results.boxes:
#         cls = int(box.cls[0])
#         if cls == 0:  # personne
#             x1, y1, x2, y2 = box.xyxy[0]
#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)
#             positions.append((cx, cy))

#     return positions

# def compute_center(positions):
#     if len(positions) == 0:
#         return None
#     x = int(np.mean([p[0] for p in positions]))
#     y = int(np.mean([p[1] for p in positions]))
#     return (x, y)

# def compute_score(center, frame_shape):
#     if center is None:
#         return 0
    
#     h, w, _ = frame_shape
#     img_center = (w // 2, h // 2)

#     dist = np.linalg.norm(np.array(center) - np.array(img_center))
#     score = max(0, 500 - dist)

#     return score

# scores = []

# for frame in frames:
#     positions = get_players_positions(frame)
#     center = compute_center(positions)
#     score = compute_score(center, frame.shape)

#     if center:
#         cv2.circle(frame, center, 10, (0, 0, 255), -1)

#     scores.append(score)

# best_index = np.argmax(scores)

# print("Meilleure caméra :", best_index + 1)

# cv2.imshow("Best Camera", frames[best_index])
# cv2.waitKey(0)
# cv2.destroyAllWindows()