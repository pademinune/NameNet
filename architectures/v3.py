
import torch
import torch.nn as nn

class Model(nn.Module):
    """~12,882 parameters"""
    def __init__(self) -> None:
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(27, 16) # 0 index is reserved for padding (empty space)
        self.layer1: nn.Linear = nn.Linear(160, 64)
        self.relu: nn.ReLU = nn.ReLU()
        self.layer2: nn.Linear = nn.Linear(64, 32)
        self.layer3: nn.Linear = nn.Linear(32, 2)
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is a size (batch_size, 10) tensor of indexes from 0 to 25 (for each character)
        does not apply softmax, instead uses crossentropy loss
        """

        x = self.embedding(x)

        x = x.flatten(1)

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)

        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Exact same as forward but applies softmax at the end - use for testing the model"""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        out = self.forward(x)

        return self.softmax(out)
    
    def predict_name(self, name: str) -> tuple[float, float]:
        inp: torch.Tensor = to_tensor(name)
        out: torch.Tensor = self.predict(inp)
        
        out = out.squeeze(dim=0)
        return (out[0].item(), out[1].item())


def to_tensor(name: str) -> torch.Tensor:
    """Returns a right-aligned tensor of the indexed characters size (10)"""
    empty_char = 0
    indexes = [empty_char for i in range(10)]
    name = name.lower()
    if (len(name) > 10):
        for i in range(1, 11):
            if (ord(name[-i]) < ord('a') or ord(name[-i]) > ord('z')):
                continue
            indexes[-i] = ord(name[-i]) - ord('a') + 1
    else:
        for i in range(1, len(name) + 1):
            if (ord(name[-i]) < ord('a') or ord(name[-i]) > ord('z')):
                continue
            indexes[-i] = ord(name[-i]) - ord('a') + 1
    return torch.tensor(indexes)

if __name__ == "__main__":
    n = Model()

    name = "justin"
    a = to_tensor(name)
    a2 = to_tensor("simon")
    print(a)
    b = torch.stack([a, a2])

    print(b.shape)

    print(b)

    print(n.predict(a))

    print(n.predict_name("justin"))
