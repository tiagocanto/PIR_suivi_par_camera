from ultralytics import YOLO
import numpy as np
import cv2

# --- CONFIG ---
IMAGE_FILES = ["portable1.jpg", "portable2.jpg", "portable3.jpg"]
RESIZE_WIDTH = 1280
MAX_PLAYER_HEIGHT = 400
MIN_PLAYERS = 3             # nombre minimum de joueurs pour qu'une caméra soit pertinente

BALL_CLASS = 0
PLAYER_CLASS = 1

# Target display size — fits a typical 1080p screen
DISPLAY_W = 1280
DISPLAY_H = 720

# couleurs des boîtes de détection
COULEUR_JOUEUR = (255, 100, 0)   # orange pour les joueurs
COULEUR_BALLE  = (0, 255, 255)   # jaune pour la balle

# --- YOLO ---
model = YOLO("yolov8n_v3_50e.pt")

# --- FONCTIONS ---
def get_detections(frame, resize_width=RESIZE_WIDTH):
    # on redimensionne l'image pour que YOLO travaille à taille fixe
    h, w = frame.shape[:2]
    scale = resize_width / w
    frame_resized = cv2.resize(frame, (resize_width, int(h * scale)))
    results = model(frame_resized)[0]

    player_positions = []  # liste des centres (cx, cy) des joueurs valides
    ball_boxes = []        # liste des boîtes de la balle (x1,y1,x2,y2)
    detections = []        # toutes les détections avec coords, classe et confiance

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])  # confiance de la détection entre 0 et 1
        x1, y1, x2, y2 = box.xyxy[0]

        # on remet les coordonnées à l'échelle de l'image originale
        x1 = int(x1 / scale); y1 = int(y1 / scale)
        x2 = int(x2 / scale); y2 = int(y2 / scale)
        height = y2 - y1

        if cls == PLAYER_CLASS and height < MAX_PLAYER_HEIGHT:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            player_positions.append((cx, cy))
            detections.append((x1, y1, x2, y2, cls, conf))
        elif cls == BALL_CLASS:
            ball_boxes.append((x1, y1, x2, y2))
            detections.append((x1, y1, x2, y2, cls, conf))

    return player_positions, ball_boxes, detections

def draw_detections(frame, detections):
    # on dessine les boîtes de détection sur l'image avec le label et la confiance
    for (x1, y1, x2, y2, cls, conf) in detections:
        if cls == PLAYER_CLASS:
            couleur = COULEUR_JOUEUR
            label = f"Joueur {conf:.0%}"
        else:
            couleur = COULEUR_BALLE
            label = f"Balle {conf:.0%}"

        # boîte autour de l'objet
        cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)

        # fond noir derrière le texte pour la lisibilité
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), couleur, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

def compute_center(positions):
    # on calcule le centre moyen de tous les joueurs détectés
    if not positions:
        return None
    return (int(np.mean([p[0] for p in positions])), int(np.mean([p[1] for p in positions])))

def compute_score(positions):
    # si pas assez de joueurs, la caméra n'est pas pertinente
    if len(positions) < MIN_PLAYERS:
        return 0

    pts = np.array(positions)

    # on calcule l'écart-type des positions x et y des joueurs
    # un écart-type faible = joueurs regroupés = il se passe quelque chose (mêlée, plaquage...)
    # un écart-type élevé = joueurs éparpillés = peu d'action concentrée
    std_x = np.std(pts[:, 0])
    std_y = np.std(pts[:, 1])
    dispersion = (std_x + std_y) / 2  # dispersion moyenne

    # score de densité : plus les joueurs sont groupés, plus le score est élevé
    score_densite = max(0, 500 - dispersion)

    # bonus proportionnel au nombre de joueurs détectés
    # plus il y a de joueurs visibles, plus la caméra est intéressante
    bonus_joueurs = len(positions) * 10

    return score_densite + bonus_joueurs

def fit_into_box(frame, box_w, box_h):
    """Resize frame to fit box while keeping aspect ratio. Letterbox with black bars."""
    h, w = frame.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    x_off = (box_w - new_w) // 2
    y_off = (box_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def draw_cell(frame, cam_idx, score, is_best, box_w, box_h):
    cell = fit_into_box(frame, box_w, box_h)
    border_color = (0, 255, 0) if is_best else (60, 60, 60)
    cv2.rectangle(cell, (0, 0), (box_w-1, box_h-1), border_color, 4 if is_best else 1)
    label = f"Cam {cam_idx+1}  |  Score: {score:.0f}" + ("  BEST" if is_best else "")
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bar_h = th + 14
    bar_col = (0, 70, 0) if is_best else (20, 20, 20)
    cv2.rectangle(cell, (0, 0), (box_w, bar_h), bar_col, -1)
    cv2.putText(cell, label, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if is_best else (160, 160, 160), 1)
    return cell

# --- LOAD IMAGES ---
frames = []
for file in IMAGE_FILES:
    frame = cv2.imread(file)
    if frame is None:
        print(f"Warning: could not load {file}")
    frames.append(frame)

if all(f is None for f in frames):
    print("Error: no images could be loaded.")
    exit(1)

# --- PROCESS ---
scores = []
for i, frame in enumerate(frames):
    if frame is None:
        scores.append(0); continue

    players, ball_boxes, detections = get_detections(frame)
    print(f"Camera {i+1}: {len(players)} joueurs détectés")

    # on dessine toutes les boîtes de détection sur l'image originale
    draw_detections(frame, detections)

    # caméra ignorée si pas assez de joueurs détectés et pas de balle
    if len(players) < MIN_PLAYERS and not ball_boxes:
        scores.append(0)
        continue

    center_players = compute_center(players)

    if ball_boxes:
        # priorité à la balle : on prend la balle la plus proche du centre de l'image
        fc = np.array([frame.shape[1]//2, frame.shape[0]//2])
        distances = [np.linalg.norm(np.array([(x1+x2)//2, (y1+y2)//2]) - fc) for x1,y1,x2,y2 in ball_boxes]
        bi = np.argmin(distances)
        # score basé sur la proximité de la balle au centre
        score = 1000 - distances[bi]
    else:
        # pas de balle détectée : on se base sur la densité des joueurs
        score = compute_score(players)

    scores.append(score)

    # point rouge au centre de masse des joueurs
    if center_players:
        cv2.circle(frame, center_players, 10, (0, 0, 255), -1)

best_idx = int(np.argmax(scores))
print(f"Best camera: Camera {best_idx+1}")
for i, s in enumerate(scores): print(f"  Camera {i+1}: {s:.1f}")

# --- BUILD GRID: 2 on top, 1 centered bottom ---
cell_w = DISPLAY_W // 2
cell_h = DISPLAY_H // 2

top_row = np.hstack([
    draw_cell(frames[0] if frames[0] is not None else np.zeros((100,100,3),dtype=np.uint8), 0, scores[0], best_idx==0, cell_w, cell_h),
    draw_cell(frames[1] if frames[1] is not None else np.zeros((100,100,3),dtype=np.uint8), 1, scores[1], best_idx==1, cell_w, cell_h),
])

# bottom: cam3 centered with black padding on sides
cam3_cell = draw_cell(frames[2] if frames[2] is not None else np.zeros((100,100,3),dtype=np.uint8), 2, scores[2], best_idx==2, cell_w, cell_h)
pad = np.zeros((cell_h, cell_w//2, 3), dtype=np.uint8)
bottom_row = np.hstack([pad, cam3_cell, pad])

grid = np.vstack([top_row, bottom_row])
cv2.imshow("Camera Grid", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
