from ultralytics import YOLO
import numpy as np
import cv2
import threading
from queue import Queue, Empty

# --- CONFIG ---
VIDEO_FILES = ["cam1.mp4", "cam2.mp4", "cam3.mp4"]
RESIZE_WIDTH = 1280          # largeur de redimensionnement pour l'inférence YOLO
FRAME_SKIP = 1               # nombre de frames à sauter entre deux inférences (1 = une sur deux)
SMOOTH_WINDOW = 5            # taille de la fenêtre de lissage temporel des scores
SWITCH_THRESHOLD = 100       # écart minimum de score pour autoriser une bascule de caméra
MAX_DISPLAY_WIDTH = 1280     # largeur maximale de la fenêtre d'affichage
MAX_PLAYER_HEIGHT = 400      # hauteur max en pixels d'un joueur — filtre les gens trop proches
MIN_PLAYERS = 3              # nombre minimum de joueurs détectés pour qu'une caméra soit pertinente

BALL_CLASS = 1               # indice de classe de la balle dans le modèle (à vérifier avec model.names)
PLAYER_CLASS = 0             # indice de classe des joueurs dans le modèle

# couleurs pour les boîtes de détection (format BGR pour OpenCV)
COULEUR_JOUEUR = (255, 100, 0)   # orange pour les joueurs
COULEUR_BALLE  = (0, 255, 255)   # jaune pour la balle

# --- CHARGEMENT DU MODÈLE ---
model = YOLO("yolov8n_v3_50e.pt")  # modèle custom entraîné sur données rugby
# vérifier les indices de classes avec : print(model.names)

# --- FONCTIONS ---

def get_detections(frame, resize_width=RESIZE_WIDTH):
    """
    Lance l'inférence YOLO sur une frame et retourne :
    - les positions (cx, cy) des joueurs valides
    - les boîtes de la balle (x1, y1, x2, y2)
    - toutes les détections avec coordonnées, classe et confiance
    """
    h, w = frame.shape[:2]
    scale = resize_width / w

    # redimensionnement pour accélérer l'inférence
    frame_resized = cv2.resize(frame, (resize_width, int(h * scale)))
    results = model(frame_resized)[0]

    player_positions = []  # centres (cx, cy) des joueurs filtrés
    ball_boxes = []        # boîtes brutes de la balle
    detections = []        # toutes les détections pour l'affichage

    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])   # confiance de la détection entre 0 et 1
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
    """
    Dessine les boîtes de détection sur la frame avec le label et la confiance.
    Couleur orange pour les joueurs, jaune pour la balle.
    """
    for (x1, y1, x2, y2, cls, conf) in detections:
        couleur = COULEUR_JOUEUR if cls == PLAYER_CLASS else COULEUR_BALLE
        label   = f"Joueur {conf:.0%}" if cls == PLAYER_CLASS else f"Balle {conf:.0%}"

        # boîte autour de l'objet détecté
        cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)

        # fond coloré derrière le texte pour la lisibilité
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), couleur, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)


def compute_center(positions):
    """
    Calcule le centroïde moyen de toutes les positions joueurs détectées.
    Retourne None si aucun joueur n'est présent.
    """
    if not positions:
        return None
    return (int(np.mean([p[0] for p in positions])),
            int(np.mean([p[1] for p in positions])))


def compute_score(positions):
    """
    Calcule le score d'une caméra basé sur la densité des joueurs.
    Utilisé uniquement quand aucune balle n'est détectée.

    Logique :
    - si moins de MIN_PLAYERS joueurs → caméra ignorée (score 0)
    - score de densité : plus les joueurs sont regroupés, plus le score est élevé
      (écart-type faible = action concentrée = mêlée, plaquage, ruck...)
    - bonus proportionnel au nombre de joueurs visibles
    """
    if len(positions) < MIN_PLAYERS:
        return 0  # pas assez de joueurs, caméra non pertinente

    pts = np.array(positions)

    # écart-type des positions x et y → mesure la dispersion des joueurs
    std_x = np.std(pts[:, 0])
    std_y = np.std(pts[:, 1])
    dispersion = (std_x + std_y) / 2

    # score de densité : maximum 500, décroît avec la dispersion
    score_densite = max(0, 500 - dispersion)

    # bonus au nombre de joueurs détectés : plus il y en a, plus c'est intéressant
    bonus_joueurs = len(positions) * 10

    return score_densite + bonus_joueurs


def show_frame(frame, window_name="Meilleure Caméra", max_width=MAX_DISPLAY_WIDTH):
    """
    Affiche une frame dans une fenêtre OpenCV.
    Redimensionne si la largeur dépasse max_width.
    """
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, frame)


# --- THREAD DE LECTURE VIDÉO ---

def video_reader(filename, queue):
    """
    Lit un fichier vidéo dans un thread dédié et pousse chaque frame dans la queue.
    Envoie None en fin de fichier pour signaler la fin du flux.
    """
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            queue.put(None)  # signal de fin de flux
            break
        queue.put(frame)
    cap.release()


# --- INITIALISATION ---

# une queue par caméra, taille max 10 pour éviter de saturer la mémoire
queues = [Queue(maxsize=10) for _ in VIDEO_FILES]

# lancement d'un thread de lecture par caméra
for i, file in enumerate(VIDEO_FILES):
    t = threading.Thread(target=video_reader, args=(file, queues[i]), daemon=True)
    t.start()

# fenêtre glissante de scores par caméra, initialisée à zéro
last_scores = [[0] * SMOOTH_WINDOW for _ in VIDEO_FILES]

current_best = 0   # indice de la caméra actuellement affichée
frame_count   = 0  # compteur global de frames pour le frame skipping

# --- BOUCLE PRINCIPALE ---

while True:
    frame_count += 1
    frames = []
    scores = []

    # on récupère une frame de chaque caméra (timeout 1s si la queue est vide)
    for q in queues:
        try:
            frame = q.get(timeout=1.0)
        except Empty:
            frame = None
        frames.append(frame)

    # si toutes les caméras ont fini, on arrête
    if all(f is None for f in frames):
        break

    # traitement de chaque caméra
    for i, frame in enumerate(frames):

        # caméra terminée ou frame manquante → on réutilise le dernier score connu
        if frame is None:
            scores.append(last_scores[i][-1])
            continue

        # frame skipping : on saute une frame sur (FRAME_SKIP + 1)
        # → réduit la charge de calcul sans trop dégrader la réactivité
        if FRAME_SKIP > 0 and frame_count % (FRAME_SKIP + 1) != 0:
            scores.append(last_scores[i][-1])
            continue

        # inférence YOLO sur la frame courante
        players, ball_boxes, detections = get_detections(frame)

        # dessin des boîtes de détection sur la frame
        draw_detections(frame, detections)

        center_players = compute_center(players)

        if ball_boxes:
            # priorité absolue à la balle
            # on sélectionne la balle la plus proche du centre de la frame
            fc = np.array([frame.shape[1] // 2, frame.shape[0] // 2])
            distances = [
                np.linalg.norm(np.array([(x1+x2)//2, (y1+y2)//2]) - fc)
                for (x1, y1, x2, y2) in ball_boxes
            ]
            best_ball_idx = np.argmin(distances)
            score = 1000 - distances[best_ball_idx]  # max 1000, toujours > score densité

        elif len(players) >= MIN_PLAYERS:
            # pas de balle mais assez de joueurs → score basé sur la densité
            score = compute_score(players)

        else:
            # pas de balle et pas assez de joueurs → caméra ignorée
            score = 0

        # mise à jour de la fenêtre glissante et calcul du score lissé
        last_scores[i].append(score)
        if len(last_scores[i]) > SMOOTH_WINDOW:
            last_scores[i].pop(0)
        smooth_score = int(np.mean(last_scores[i]))
        scores.append(smooth_score)

        # point rouge au centroïde des joueurs pour le débogage visuel
        if center_players:
            cv2.circle(frame, center_players, 10, (0, 0, 255), -1)

    # --- DÉCISION DE BASCULE ---

    new_best = np.argmax(scores)

    # on ne bascule que si la nouvelle caméra dépasse l'actuelle d'au moins SWITCH_THRESHOLD
    # → évite les oscillations entre deux caméras aux scores proches
    if new_best != current_best:
        if scores[new_best] > scores[current_best] + SWITCH_THRESHOLD:
            current_best = new_best

    # --- AFFICHAGE ---

    best_frame = frames[current_best]
    if best_frame is not None:
        # étiquette de la caméra sélectionnée + son score lissé
        label = f"Camera {current_best + 1}  |  Score: {scores[current_best]}"
        cv2.putText(best_frame, label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        show_frame(best_frame)

    # appui sur Échap pour quitter
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
