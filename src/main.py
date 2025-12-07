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
    max_num_hands=1,
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

    # extract reton of interest where fruit will be drawn
    roi = frame[y : y + size, x : x + size]
    if (
        roi.shape[0] != img_resized.shape[0]
        or roi.shape[1] != img_resized.shape[1]
    ):
        img_resized = cv2.resize(img_resized, (roi.shape[1], roi.shape[0]))
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))

    roi[:] = (img_resized[:, :, :3] * mask + roi * (1 - mask)).astype(np.uint8)


# game loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # spawn fruits
    if time.time() - last_spawn_time > spawn_rate:
        fruits.append(spawn_fruit(w, h))
        last_spawn_time = time.time()

    # update fruits
    for fruit in fruits:
        fruit["x"] += fruit["vx"]
        fruit["y"] += fruit["vy"]
        fruit["vy"] += GRAVITY

    # remove off-screen fruits (apply miss penalty)
    remaining_fruits = []
    for fruit in fruits:
        if fruit["y"] - fruit["radius"] > h:
            score -= miss_penalty
        else:
            remaining_fruits.append(fruit)
    fruits = remaining_fruits

    # process hand landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    finger_pos = None
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        index_finger_tip = hand_landmarks.landmark[8]
        finger_x = int(index_finger_tip.x * w)
        finger_y = int(index_finger_tip.y * h)
        finger_pos = (finger_x, finger_y)

        # draw fingertip
        cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)

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

    cv2.imshow("Fruit Ninja Webcam Game", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
