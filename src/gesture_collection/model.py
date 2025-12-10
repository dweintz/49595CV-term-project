import torch.nn as nn

class GestureModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_landmarks = 43  # 21 left + 21 right + wrist delta
        self.conv_block = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, self.num_landmarks, 3)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = self.mlp_head(x)
        return x