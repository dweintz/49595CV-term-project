import cv2
import mediapipe as mp
import os
import csv
import time
import numpy as np

# file config
OUT_DIR = "gesture_dataset"
IMG_DIR = os.path.join(OUT_DIR, "images")
CSV_PATH = os.path.join(OUT_DIR, "labels.csv")

LABEL_KEYS = {
    '1': 'pause',
    '2': 'restart',
    '3': 'quit',
    '0': 'none',
    'q': 'quit_program'
}

TARGET_PER_CLASS = 250  # max number of samples per class
CAPTURE_COOLDOWN = 0.5  # wait time in seconds between pictures
MIN_DET_CONF = 0.5  # minimum hand detection confidence

os.makedirs(IMG_DIR, exist_ok=True)

# setup CSV file for storing annotations
if not os.path.exists(CSV_PATH):
    header = (
        ["label", "image_path"] +
        [f"left_{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")] +
        [f"right_{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")] +
        ["dx", "dy", "dz"]
    )
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(header)

# load existing counts for each label
sample_counts = {lbl: 0 for lbl in LABEL_KEYS.values() if lbl != "quit_program"}

if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row["label"]
            if lbl in sample_counts:
                sample_counts[lbl] += 1

print("\nCurrent sample counts:")
for lbl, count in sample_counts.items():
    print(f"  {lbl}: {count}")
print("\n")

# initialize MediaPipe Hands
mp_hands = mp.solutions.hands  # type: ignore[attr-defined]

# open webcam
cap = cv2.VideoCapture(0)

last_capture_time = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=0.5
) as hands:

    current_label = None

    # annotation tool controls
    print("\nGesture Collector for Pause / Restart / Quit / None\n")
    print("Controls:")
    print("  '1' → pause")
    print("  '2' → restart")
    print("  '3' → quit")
    print("  '0' → none")
    print("SPACE → capture")
    print("  'q' → quit program\n")

    while True:
        # read webcam frame
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display = frame.copy()

        # draw detected hands
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(  # type: ignore[attr-defined]
                    display, hl, mp_hands.HAND_CONNECTIONS
                )

        # display current label + progress counts
        if current_label:
            progress = f"{sample_counts[current_label]}/{TARGET_PER_CLASS}"
            text = f"Label: {current_label}   {progress}"
        else:
            text = "Label: None"

        cv2.putText(display, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Dataset Collector", display)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            c = chr(key)

            # user switches label
            if c in LABEL_KEYS:
                if LABEL_KEYS[c] == "quit_program":
                    break
                current_label = LABEL_KEYS[c]
                print(f"Selected -> {current_label}")
                continue

            # SPACE: capture sample (key 32)
            if key == 32:  
                now = time.time()

                # enforce 0.5 second cooldown
                if now - last_capture_time < CAPTURE_COOLDOWN:
                    wait_left = CAPTURE_COOLDOWN - (now - last_capture_time)
                    print(f"Wait {wait_left:.2f}s...")
                    continue
                
                # ensure a label is selected
                if current_label is None:
                    print("Select a label first!")
                    continue

                # enforce class limit
                if sample_counts[current_label] >= TARGET_PER_CLASS:
                    print(f"Limit of {TARGET_PER_CLASS} reached for '{current_label}'.")
                    continue

                # NONE class does not require hands
                if current_label == "none":
                    left_lm = np.zeros((21, 3))
                    right_lm = np.zeros((21, 3))
                    wrist_diff = np.zeros(3)

                else:
                    # need exactly 2 hands
                    if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
                        print("Need TWO hands visible for this gesture.")
                        continue

                    handedness = results.multi_handedness
                    lh, rh = None, None

                    # assign left/right
                    for hl, hd in zip(results.multi_hand_landmarks, handedness):
                        if hd.classification[0].label == "Left":
                            lh = hl
                        else:
                            rh = hl

                    if lh is None or rh is None:
                        print("Both hands must be visible.")
                        continue

                    # normalize
                    def process_hand(hand):
                        lm = np.array([[l.x, l.y, l.z] for l in hand.landmark])
                        wrist = lm[0].copy()
                        lm -= wrist
                        scale = np.max(np.linalg.norm(lm[:, :2], axis=1))
                        if scale < 1e-6:
                            scale = 1.0
                        lm /= scale
                        return lm, wrist

                    left_lm, left_wrist = process_hand(lh)
                    right_lm, right_wrist = process_hand(rh)
                    wrist_diff = right_wrist - left_wrist

                # save image
                name = f"{current_label}_{int(time.time()*1000)}.jpg"
                img_path = os.path.join(IMG_DIR, name)
                cv2.imwrite(img_path, frame)

                # save CSV row
                row = (
                    [current_label, img_path] +
                    left_lm.flatten().tolist() +
                    right_lm.flatten().tolist() +
                    wrist_diff.tolist()
                )

                with open(CSV_PATH, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                # increment count
                sample_counts[current_label] += 1
                last_capture_time = now

                print(f"Saved sample → {img_path}   ({sample_counts[current_label]}/{TARGET_PER_CLASS})")

cap.release()
cv2.destroyAllWindows()
print("Done.")
