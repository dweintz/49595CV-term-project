"""
Game prototype
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import pygame

GAME_W, GAME_H = 640, 400  # size of webcam game window
HAND_FRAME_SKIP = 2  # run hand detection every N frames
DETECT_W, DETECT_H = 320, 180  # resolution of hand detection frame
GRAVITY = 0.28  # gravity (how fast fruits fall)

BACKGROUND_MUSIC = "assets/background.mp3"
SLICE_SOUND = "assets/slash-21834.wav"

# initialize MediaPipe Hands
mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    model_complexity=0,
)
mp_draw = mp.solutions.drawing_utils  # type: ignore[attr-defined]

# open webcam
cv2.setUseOptimized(True)
cv2.setNumThreads(4)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, GAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GAME_H)

# initialize game sound effects
pygame.mixer.init()
slice_sound = pygame.mixer.Sound(SLICE_SOUND)
slice_sound.set_volume(1.0)

# initialize game background music
pygame.mixer.music.load(BACKGROUND_MUSIC)
pygame.mixer.music.set_volume(0.3)

# loop background music indefinitely
pygame.mixer.music.play(-1)

# load fruit images
fruit_images = [
    cv2.imread(f"assets/fruit/fruit{i}.png", cv2.IMREAD_UNCHANGED)
    for i in range(1, 3)
]
PRECOMPUTED_SIZES = {size: [] for size in (40, 80, 120)}
for img in fruit_images:
    for size in PRECOMPUTED_SIZES:
        PRECOMPUTED_SIZES[size].append(cv2.resize(img, (size, size), cv2.INTER_AREA))  # type: ignore[arg-type]

# game variables
fruits = []
spawn_rate = 0.8
last_spawn_time = time.time()
score = 0
miss_penalty = 1
paused = False


def spawn_fruit(frame_width, frame_height):
    """
    Randomly spawn fruit on the frame.

    :param frame_width: width of frame
    :param frame_height: height of frame
    """

    # choose random size
    size = random.choice([40, 80, 120])
    radius = size // 2

    # choose random position
    x = random.randint(radius, frame_width - radius)
    y = random.randint(frame_height // 2, frame_height - radius)

    # choose velocity
    vx = random.uniform(-3, 3)
    vy = -random.uniform(7, 12)

    # choose random fruit
    img = random.choice(fruit_images)

    return {"x": x, "y": y, "radius": radius, "vx": vx, "vy": vy, "img": img}


def check_collision(fruit, finger_x, finger_y):
    """
    Check for collision between finger and fruit.

    :param fruit: dictionary of fruit characteristics
    :param finger_x: x-location of finger
    :param finger_y: y-location of finger
    """
    distance = math.hypot(fruit["x"] - finger_x, fruit["y"] - finger_y)
    return distance < fruit["radius"]


def draw_fruit(frame, fruit):
    """
    Draw fruit on the image.

    :param frame: Webcam frame
    :param fruit: dictionary of fruit characteristics
    """

    # open and resize fruit image to radius
    img = fruit["img"]
    size = fruit["radius"] * 2
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # create circular mask
    alpha = img_resized[:, :, 3] if img_resized.shape[2] == 4 else 255
    mask = cv2.merge([alpha, alpha, alpha])  # type: ignore
    mask = mask / 255.0

    # compute fruit location on frame
    x = int(fruit["x"] - fruit["radius"])
    y = int(fruit["y"] - fruit["radius"])
    h, w = frame.shape[:2]

    # Check bounds
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + size > w:
        size = w - x
    if y + size > h:
        size = h - y

    # extract region of interest where fruit will be drawn
    roi = frame[y : y + size, x : x + size]

    # skip if roi has zero width or height
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return

    if (
        roi.shape[0] != img_resized.shape[0]
        or roi.shape[1] != img_resized.shape[1]
    ):
        img_resized = cv2.resize(img_resized, (roi.shape[1], roi.shape[0]))
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))

    roi[:] = (img_resized[:, :, :3] * mask + roi * (1 - mask)).astype(np.uint8)


def hand_orientation(landmarks, w, h):
    """
    Get orientation of hands for game pause/play

    :param landmarks: hand landmarks from mediapipe.
    :param w: width of frame
    :param h: height of frame
    """

    # wrist coordinates
    wrist = landmarks.landmark[0]

    # index finger coordinates
    idx_mcp = landmarks.landmark[5]

    # compute horizonal and vertial distances between wrist and index finger
    dx = abs(idx_mcp.x - wrist.x) * w
    dy = abs(idx_mcp.y - wrist.y) * h

    # decide hand orientation based on x-y distance ratios
    if dx > dy * 1.2:
        return "horizontal"
    elif dy > dx * 1.2:
        return "vertical"
    else:
        return "unknown"


def hands_touching(handA, handB, w, h):
    """
    Check if two hands are touching.

    :param handA: MediaPipe Hands landmarks for hand A
    :param handB: MediaPipe Hands landmarks for hand B
    :param w: width of frame
    :param h: height of frame
    """

    # compute center of each hand
    ax = np.mean([lm.x for lm in handA.landmark]) * w
    ay = np.mean([lm.y for lm in handA.landmark]) * h
    bx = np.mean([lm.x for lm in handB.landmark]) * w
    by = np.mean([lm.y for lm in handB.landmark]) * h

    # approximate hand size by distance between wrist and middle finger MCP
    handA_size = math.hypot(
        (handA.landmark[9].x - handA.landmark[0].x) * w,
        (handA.landmark[9].y - handA.landmark[0].y) * h,
    )
    handB_size = math.hypot(
        (handB.landmark[9].x - handB.landmark[0].x) * w,
        (handB.landmark[9].y - handB.landmark[0].y) * h,
    )

    # dynamic threshold: sum of half hand sizes
    threshold = (handA_size + handB_size) * 0.6

    distance = math.hypot(ax - bx, ay - by)
    return distance < threshold


def draw_pause_overlay(frame):
    """
    Draw pause screen on video frame.

    :param frame: video frame
    """

    # copy frame
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # darken overlay
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)

    # blend with transparency
    alpha = 0.55
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # draw paused text on frame
    text = "PAUSED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    thickness = 4
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]

    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2

    cv2.putText(
        frame, text, (text_x, text_y), font, scale, (255, 255, 255), thickness
    )


pause_gesture_frames = 0  # how many frames pause gesture has been detected
PAUSE_GESTURE_FRAMES_REQUIRED = 3  # gesture must be held for 3 frames
pause_cooldown = 3.0  # seconds to wait before detecting next pause/unpause
last_pause_time = 0.0  # last time pause/unpause occurred

# game loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    current_time = time.time()

    # process hand landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    finger_pos = None

    # if hands are detected
    if result.multi_hand_landmarks:
        landmarks_list = result.multi_hand_landmarks

        # draw all hands
        for lm in landmarks_list:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # pause gesture detection (needs exactly 2 hands)
        if len(landmarks_list) == 2:
            handA, handB = landmarks_list
            oA = hand_orientation(handA, w, h)
            oB = hand_orientation(handB, w, h)

            # only check gesture if cooldown has passed
            if current_time - last_pause_time > pause_cooldown:
                if hands_touching(handA, handB, w, h) and {
                    "horizontal",
                    "vertical",
                } == {oA, oB}:
                    pause_gesture_frames += 1
                    if pause_gesture_frames >= PAUSE_GESTURE_FRAMES_REQUIRED:
                        paused = not paused
                        last_pause_time = current_time
                        pause_gesture_frames = 0
                else:
                    pause_gesture_frames = 0
        else:
            pause_gesture_frames = 0

        # track index finger of first hand only if not paused
        if not paused:
            hand_landmarks = landmarks_list[0]
            index_finger_tip = hand_landmarks.landmark[8]
            finger_x = int(index_finger_tip.x * w)
            finger_y = int(index_finger_tip.y * h)
            finger_pos = (finger_x, finger_y)
            cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)

    # update fruits if not paused
    if not paused:
        # spawn fruits
        if time.time() - last_spawn_time > spawn_rate:
            fruits.append(spawn_fruit(w, h))
            last_spawn_time = time.time()

        # update fruit positions
        for fruit in fruits:
            fruit["x"] += fruit["vx"]
            fruit["y"] += fruit["vy"]
            fruit["vy"] += GRAVITY

        # remove off-screen fruits and apply miss penalty
        remaining_fruits = []
        for fruit in fruits:
            if fruit["y"] - fruit["radius"] > h:
                score -= miss_penalty
            else:
                remaining_fruits.append(fruit)
        fruits = remaining_fruits

    # check collisions
    if finger_pos:
        new_fruits = []
        for fruit in fruits:
            if check_collision(fruit, finger_pos[0], finger_pos[1]):
                score += 1
                slice_sound.play()
            else:
                new_fruits.append(fruit)
        fruits = new_fruits

    # draw fruits
    for fruit in fruits:
        draw_fruit(frame, fruit)

    # draw score
    cv2.putText(
        frame,
        f"Score: {score}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )

    # draw pause overlay if paused
    if paused:
        draw_pause_overlay(frame)

    # show frame
    cv2.imshow("Fruit Ninja Webcam Game", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
