
import torch
import torch.nn as nn
from to_vector import to_tensor

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1: nn.Linear = nn.Linear(26, 20)
        self.activator1: nn.ReLU = nn.ReLU()
        self.layer2: nn.Linear = nn.Linear(20, 2)
        # self.layer3 = nn.Linear(20, 2)
        self.activator2: nn.Softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activator1(x)
        x = self.layer2(x)
        # x = self.activator1(x)
        # x = self.layer3(x)
        return self.activator2(x)
    
    def predict(self, name: str) -> str:
        out = self.forward(to_tensor(name))
        index = torch.argmax(out)
        if index.item() == 0:
            return "Male"
        else:
            return "Female"
    
    def __str__(self) -> str:
        return f"26 -> 16 -> ReLu -> 2 -> Softmax"


def to_tensor(name: str) -> torch.Tensor:
    tensor: torch.Tensor = torch.zeros(26)

    for char in name:
        char = char.lower()
        index: int = ord(char) - ord('a')

        tensor[index] += 1

    return tensor

