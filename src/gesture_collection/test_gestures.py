import cv2
import torch

from src.gesture_collection.model import GestureModel
from src.utils import load_mediapipe
from src.utils import extract_features

CONF_THRESHOLD = 0.9
MIN_HAND_CONFIDENCE = 0.6

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

# open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    fv = extract_features(results)

    if fv is None:
        label = "none"
    else:
        x = torch.tensor(fv, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            conf = conf.item()
            pred = pred.item()

            if conf < CONF_THRESHOLD:
                label = "none"
            else:
                label = class_names[pred]

    # draw label
    cv2.rectangle(frame, (0, 0), (260, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture: {label}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Gesture Tester", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
