import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

from src.gesture_collection.model import GestureModel

CSV_PATH = "gesture_dataset/labels.csv"
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read CSV file with data labels
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} samples.")

left_cols  = [f"left_{i}_{a}"  for i in range(21) for a in ("x","y","z")]
right_cols = [f"right_{i}_{a}" for i in range(21) for a in ("x","y","z")]
delta_cols = ["dx", "dy", "dz"]
feature_cols = left_cols + right_cols + delta_cols

df = df.dropna(subset=feature_cols).reset_index(drop=True)

le = LabelEncoder()
df["label_idx"] = le.fit_transform(df["label"])
NUM_CLASSES = len(le.classes_)
INPUT_DIM = len(feature_cols)

class GestureDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row[feature_cols].astype(float).values.astype(np.float32)
        y = int(row["label_idx"])
        return x, y

# instantiate dataset class
dataset = GestureDataset(df)
n = len(dataset)
n_val = max(20, int(0.2 * n))
train_ds, val_ds = random_split(dataset, [n-n_val, n_val])

# load training and validation data
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# gesture model
model = GestureModel(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# training loop
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)

    avg_loss = train_loss / len(train_loader.dataset)  # type: ignore[attr-defined]

    # validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            out = model(xb)
            pred = out.argmax(1)

            correct += (pred == yb).sum().item()
            total += xb.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch:02} | loss={avg_loss:.4f} | val_acc={val_acc:.3f}")

# save model
torch.save({
    "model": model.state_dict(),
    "classes": list(le.classes_)
}, "gesture_mlp.pth")

print("Saved model -> gesture_mlp.pth")
