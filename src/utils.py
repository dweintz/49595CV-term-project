import cv2
import mediapipe as mp
import numpy as np
import random
import math
from PIL import Image

SCORE_PATH = "C:/Users/Donny Weintz/Downloads/ECE 49595CV/49595CV-term-project/assets/game_stats.txt"


def load_mediapipe(min_confidence):
    """
    Initialize MediaPipe Hands model.
    
    :param min_confidence: Min confidence for hand detections
    """

    # initialize MediaPipe Hands
    mp_hands = mp.solutions.hands  # type: ignore[attr-defined]

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_confidence,
        model_complexity=0,
    )

    mp_draw = mp.solutions.drawing_utils  # type: ignore[attr-defined]

    return mp_hands, hands, mp_draw


def extract_features(results):
    """
    Takes two hands from MediaPipe and turns them into feature
    vectos for gesture detection model.
    
    :param results: MediaPipe hands results
    """

    # if not two hands, return None
    if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
        return None
    
    # separate left and right hands
    left, right = None, None
    for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
        if handed.classification[0].label == "Left":
            left = lm
        else:
            right = lm
    if left is None or right is None:
        return None

    # convert 21 hand landmarks into array (21, 3)
    def normalize(hand):
        wrist = np.array([hand.landmark[0].x, hand.landmark[0].y, hand.landmark[0].z])
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        pts -= wrist
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist < 1e-6:
            return None
        pts /= max_dist
        return pts.flatten()

    # apply normalization to left and right
    l = normalize(left)
    r = normalize(right)
    if l is None or r is None:
        return None

    # compute realtive wrist positions
    delta = np.array([right.landmark[0].x - left.landmark[0].x,
                      right.landmark[0].y - left.landmark[0].y,
                      right.landmark[0].z - left.landmark[0].z])
    
    # create final feature vector
    feat = np.concatenate([l, r, delta]).astype(np.float32)
    return feat.reshape(1, -1)


def load_gif(gif):
    """
    Function to load a GIF
    
    :param gif: gif file
    """

    gif = Image.open(gif)
    frames = []
    try:
        while True:
            frame = gif.convert("RGBA")
            frame = np.array(frame)
            frames.append(frame)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    return frames


def spawn_fruit(frame_width, frame_height, fruit_images, bomb_probability):
    """
    Randomly spawn fruit on the frame.

    :param frame_width: width of frame
    :param frame_height: height of frame
    """

    # choose random size
    size = random.choice([60, 80, 110])
    radius = size // 2

    # spawn at bottom edge
    x = random.randint(radius + int(frame_width * 0.15), frame_width - radius - int(frame_width * 0.15))
    y = frame_height + radius   

    # upward velocity (negative)
    vx = random.uniform(-3, 3)
    vy = -random.uniform(10, 15) 

    # decide whether fruit is bomb
    if random.random() < bomb_probability:
        img_file = "assets/bomb.png"
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (80, 80))
        radius = 40

        return {
            "x": x,
            "y": y,
            "radius": radius,
            "vx": vx,
            "vy": vy,
            "img": img,
            "type": "bomb",
            "frame": 0,
            "exploding": False
        }
    else:
        img = random.choice(fruit_images)
        return {
            "x": x,
            "y": y,
            "radius": radius,
            "vx": vx,
            "vy": vy,
            "img": img,
            "type": "fruit",
            "frame": 0,
            "exploding": False
        }


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


def draw_game_overlay(frame, text, score):
    """
    Draw overlay on screen.
    
    :param frame: frame
    :param text: text
    """

    # copy frame
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # darken overlay
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)

    # blend with transparency
    alpha = 0.55
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # draw overlay text on frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    thickness = 4
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]

    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2 - 35

    color = (0, 0, 255)
    if text == "PAUSED":
        color = (255, 255, 255)
    
    cv2.putText(
        frame, text, (text_x, text_y), font, scale, color, thickness
    )

    # place extra text for game over screen
    if text == "GAME OVER":
        high_score = load_high_score(SCORE_PATH)
        if score > high_score:
            high_score = score
            save_high_score(score, SCORE_PATH)
        high_score_text = f"High Score: {high_score}"
        cv2.putText(
            frame, high_score_text, (50, text_y + 50), font, 0.75, (255, 255, 255), 2
        )
        instruction_text = "Double Thumbs up to restart game"
        cv2.putText(
            frame, instruction_text, (50, text_y + 100), font, 0.75, (255, 255, 255), 2
        )    
        instruction_text = "Double Thumbs down to restart quit"
        cv2.putText(
            frame, instruction_text, (50, text_y + 150), font, 0.75, (255, 255, 255), 2
        )  

def load_high_score(path):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0 
    

def save_high_score(score, path):
    with open(path, "w") as f:
        f.write(str(score))