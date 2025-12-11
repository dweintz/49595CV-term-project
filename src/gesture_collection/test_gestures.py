import cv2
import torch

from src.gesture_collection.model import GestureModel
from src.utils import load_mediapipe, extract_features

CONF_THRESHOLD = 0.9
MIN_HAND_CONFIDENCE = 0.6

# load the saved gesture model checkpoint
checkpoint = torch.load("gesture_mlp.pth", map_location="cpu")
class_names = checkpoint["classes"]
num_classes = len(class_names)

# load architecture, weights, and set model to inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureModel(input_dim=129, hidden_dim=256, num_classes=num_classes).to(device)
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

    label = "none"

    if results.multi_hand_landmarks:
        # draw landmarks
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # extract features for gesture
        fv = extract_features(results)  # returns (1, 129)
        if fv is not None:
            x = torch.tensor(fv, dtype=torch.float32).to(device)
            x = x.squeeze(0)  # remove batch dimension -> shape (129,)
            
            with torch.no_grad():
                logits = model(x.unsqueeze(0))  # add batch dim for MLP -> shape (1, num_classes)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)

                conf = conf.item()
                pred = pred.item()
                if conf >= CONF_THRESHOLD:
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
