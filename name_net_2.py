
import torch
import torch.nn as nn

class NameNet2(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(27, 16) # 0 index is reserved for padding (empty space)
        self.layer1: nn.Linear = nn.Linear(160, 30)
        self.relu: nn.ReLU = nn.ReLU()
        # self.layerm: nn.Linear = nn.Linear(30, 30)
        self.layer2: nn.Linear = nn.Linear(30, 2)
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is a size (batch_size, word_size) tensor of indexes from 0 to 25 (for each character)
        does not apply softmax, instead uses crossentropy loss
        """
        # try:
        x = self.embedding(x)
        # except:
        #     print("FAILED ON EMBEDDING")
        #     print(x)
        #     exit()
        x = x.flatten(1)

        x = self.layer1(x)
        x = self.relu(x)

        # x = self.layerm(x)
        # x = self.relu(x)

        x = self.layer2(x)
        # x = self.softmax(x)

        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Exact same as forward but applies softmax at the end - use for testing the model"""
        return self.softmax(self.forward(x))

if __name__ == "__main__":
    n = NameNet2()
    a = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

    print(n.embedding(a))
    print(n(a))

    from to_vector import index_tensor

    v = index_tensor("alex")
    w = index_tensor("philza")
    # print(torch.unsqueeze(torch.tensor(2), 0))
    print(v)
    t = torch.stack([v,w])
    print(t.shape)
    print(n(t))
