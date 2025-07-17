
import torch
import torch.nn as nn



class RecurrentNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(27, 16)
        self.rnn: nn.RNN = nn.RNN(input_size = 16, hidden_size = 20, batch_first=True)

        self.fc = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()
        # x is a list of letter indexes from start to end
        x = self.embedding(x) # shape word_length, 16
        # print(x)
        output, h_n = self.rnn(x)
        # print(output) # prints all the hidden outputs
        h_n = h_n.squeeze()
        # print(h_n) # prints only the last hidden output
        

        final = self.fc(h_n)
        return final


def to_tensor(name: str) -> torch.Tensor:
    name = name.lower()
    lst: list[int] = []
    for c in name:
        n: int = ord(c) - ord('a') + 1
        if n >= 1 and n <= 26:
            lst.append(n)
        else:
            lst.append(0)
    return torch.tensor(lst)

if __name__ == "__main__":
    m = RecurrentNet()

    print(m(torch.tensor([0])))
    print(m(torch.tensor([2, 3, 5, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])))

    print(to_tensor("justin"))
