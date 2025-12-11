import torch.nn as nn
import torch.nn.functional as F

class GestureModel(nn.Module):
    def __init__(self, input_dim=129, hidden_dim=256, num_classes=10):
        super(GestureModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.mlp_head(x)
        return x
