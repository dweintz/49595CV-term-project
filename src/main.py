import cv2
import numpy as np
import time
import pygame
import torch

from utils import load_mediapipe
from utils import extract_features
from utils import load_gif
from utils import spawn_fruit
from utils import check_collision
from utils import draw_fruit
from utils import draw_game_overlay
from gesture_collection.model import GestureModel

GAME_W, GAME_H = 640, 400  # size of webcam game window
HAND_FRAME_SKIP = 2  # run hand detection every N frames
MIN_HAND_CONFIDENCE = 0.7  # minimum confidence for hand detection
DETECT_W, DETECT_H = 320, 180  # resolution of hand detection frame
GRAVITY = 0.28  # gravity (how fast fruits fall)
BOMB_PROBABILITY = 0.12  # probability of spawning a bomb
MIN_GESTURE_CONFIDENCE = 0.85  # minimum confidence for gesture recognition
EXPLOSION_DURATION = 3  # explosion duration in seconds
GESTURE_FRAMES_REQUIRED = 3  # number of frames required to count gesture
SHOW_HANDS = False  # whether to show hands annotations during gameplay

BACKGROUND_MUSIC = "assets/music.mp3"
SLICE_SOUND = "assets/sword.mp3"
EXPLOSION_SOUND = "assets/explosion.mp3"
EXPLOSION_GIF = "assets/explosion_animation.gif"

# load the saved gesture model checkpoint
checkpoint = torch.load("gesture_mlp.pth", map_location="cpu")
class_names = checkpoint["classes"]
num_classes = len(class_names)

# load architecture, weights, and set model to inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureModel(num_classes)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

model = GestureModel(num_classes)
model.load_state_dict(checkpoint["model"])
model.eval()

# initialize MediaPipe Hands
mp_hands, hands, mp_draw = load_mediapipe(MIN_HAND_CONFIDENCE)

# load explosion GIF frames
explosion_frames = load_gif(EXPLOSION_GIF)

# open webcam
cv2.setUseOptimized(True)
cv2.setNumThreads(4)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, GAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GAME_H)

# initialize slice sound effect
pygame.mixer.init()
slice_sound = pygame.mixer.Sound(SLICE_SOUND)
slice_sound.set_volume(1.0)

# initialize explosion sound effect
explosion_sound = pygame.mixer.Sound(EXPLOSION_SOUND)
explosion_sound.set_volume(1.0)

# initialize game background music
pygame.mixer.music.load(BACKGROUND_MUSIC)
pygame.mixer.music.set_volume(0.3)

# loop background music indefinitely
pygame.mixer.music.play(-1)

# load fruit images
fruit_images = [
    cv2.imread(f"assets/fruit/fruit{i}.png", cv2.IMREAD_UNCHANGED)
    for i in range(1, 9)
]
PRECOMPUTED_SIZES = {size: [] for size in (60, 80, 110)}
for img in fruit_images:
    for size in PRECOMPUTED_SIZES:
        PRECOMPUTED_SIZES[size].append(cv2.resize(img, (size, size), cv2.INTER_AREA))  # type: ignore[arg-type]

# function to quit the game
def quit_game():
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    exit()

def run_game():
    global cap
    global fruit_images
    global slice_sound
    global explosion_sound
    global explosion_gif

    fruits = []  # list storing all fruits present on screen
    spawn_rate = 0.8  # seconds between fruit spawns
    last_spawn_time = 0  # timestamp of last fruit spawn
    score = 0  # current score
    max_score = 0  # total fruits that have appeared
    miss_penalty = 1  # score penalty when fruit is missed
    overlay = False  # True when game is paused

    gesture_frames = 0  # counts consecutive frames with same gesture
    gesture_cooldown = 1.5  # time in seconds required before next gesture activates
    last_gesture_time = 0.0  # timestamp of last accepted gesture

    game_over = False  # becomes true when player loses
    explosion_playing = False  # True when explosion GIF is currently showing
    explosion_start_time = 0  # timestamp of when explosion animation started

    while True:
        # read a frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        current_time = time.time()

        # process hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        finger_pos = None
        gesture_label = None

        # only run gesture classification if 2 hands are visible
        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
            fv = extract_features(result)
            if fv is not None:
                x = torch.tensor(fv, dtype=torch.float32)

                # run gesture detection model
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = torch.max(probs, dim=1)

                    # only accept gestures above confidence threshold
                    if conf.item() >= MIN_GESTURE_CONFIDENCE:
                        gesture_label = class_names[pred.item()]

        # process gestures
        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
            if gesture_label:
                # restart gesture
                if gesture_label == "restart":
                    gesture_frames += 1
                    if gesture_frames >= GESTURE_FRAMES_REQUIRED and current_time - last_gesture_time > gesture_cooldown:
                        # reset game variables
                        fruits = []
                        score = 0
                        max_score = 0
                        game_over = False
                        explosion_playing = False
                        explosion_start_time = None
                        overlay = False
                        last_gesture_time = current_time
                        gesture_frames = 0

                # pause gesture
                elif gesture_label == "pause":
                    gesture_frames += 1
                    if gesture_frames >= GESTURE_FRAMES_REQUIRED and current_time - last_gesture_time > gesture_cooldown:
                        # toggle pause overlay
                        overlay = not overlay
                        last_gesture_time = current_time
                        gesture_frames = 0
                
                # quit gesture
                elif gesture_label == "quit":
                    quit_game()
                
                else:
                    gesture_frames = 0
            else:
                gesture_frames = 0

        # draw hands and fingertip
        if result.multi_hand_landmarks:
            if SHOW_HANDS:
                for lm in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # only track slicing finger when game not paused or over
            if not overlay and not game_over:
                hand_landmarks = result.multi_hand_landmarks[0]
                index_tip = hand_landmarks.landmark[8]
                finger_x = int(index_tip.x * w)
                finger_y = int(index_tip.y * h)
                finger_pos = (finger_x, finger_y)

                # draw blue dot in fingertip
                cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)

        # game logic
        if not game_over and not explosion_playing:
            if not overlay:
                # spawn next fruit
                if current_time - last_spawn_time > spawn_rate:
                    fruits.append(spawn_fruit(w, h, fruit_images, BOMB_PROBABILITY))
                    last_spawn_time = current_time

                # update fruit positions
                for fruit in fruits:
                    fruit["x"] += fruit["vx"]
                    fruit["y"] += fruit["vy"]
                    fruit["vy"] += GRAVITY

                # check missed fruits
                remaining_fruits = []
                for fruit in fruits:
                    if fruit["y"] - fruit["radius"] > h and fruit["type"] != "bomb":
                        score -= miss_penalty
                        if max_score - score > 3:
                            game_over = True
                        continue
                    remaining_fruits.append(fruit)
                fruits = remaining_fruits

                # check slicing collisions
                if finger_pos:
                    new_fruits = []
                    for fruit in fruits:
                        if check_collision(fruit, finger_pos[0], finger_pos[1]):
                            # bomb slice -> trigger game over
                            if fruit["type"] == "bomb":
                                explosion_sound.play()
                                explosion_start_time = time.time()
                                explosion_playing = True
                                game_over = True
                            
                            # increase slice score
                            else:
                                score += 1
                                max_score += 1
                                slice_sound.play()
                        else:
                            new_fruits.append(fruit)
                    fruits = new_fruits

        # draw all active fruits
        for fruit in fruits:
            if fruit["type"] == "fruit":
                draw_fruit(frame, fruit)
            elif fruit["type"] == "bomb":
                draw_fruit(frame, fruit) 

        # draw score and strikes
        cv2.putText(frame, f"Score: {score}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"Strikes: {max_score - score}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # draw overlay
        if overlay:
            draw_game_overlay(frame, "PAUSED", score)

        # play explosion GIF if a bomb was hit
        if explosion_playing:
            # ensure valid start time
            if explosion_start_time is None:
                explosion_start_time = time.time()

            elapsed = time.time() - explosion_start_time 
            frame_idx = int(elapsed / EXPLOSION_DURATION * len(explosion_frames))
            frame_idx = min(frame_idx, len(explosion_frames) - 1)

            gif_frame = explosion_frames[frame_idx]
            gif_frame = cv2.resize(gif_frame, (w, h), interpolation=cv2.INTER_AREA)

            gif_bgr = cv2.cvtColor(gif_frame, cv2.COLOR_RGBA2BGR)
            alpha = gif_frame[:, :, 3] / 255.0  # keep alpha channel

            # alpha-blend GIF onto webcam
            for c in range(3):
                frame[:, :, c] = (gif_bgr[:, :, c] * alpha + frame[:, :, c] * (1 - alpha)).astype(np.uint8)

            # stop when GIF finishes
            if elapsed >= EXPLOSION_DURATION:
                explosion_playing = False  

        # draw game over overlay if not playing explosion
        if game_over and not explosion_playing:
            draw_game_overlay(frame, "GAME OVER", score)

        # quit game via 'q' key
        cv2.imshow("Fruit Ninja Webcam Game", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            quit_game()


while True:
    run_game()
