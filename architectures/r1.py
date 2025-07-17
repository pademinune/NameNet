
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence



class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(27, 16, padding_idx=0)
        self.rnn: nn.RNN = nn.RNN(input_size = 16, hidden_size = 128, batch_first=True)

        self.fc = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is a list of letter indexes from start to end.
        size (batches, 10)
        """
        # x = x.int()

        x = self.embedding(x) # shape (batches, 10, 16)
        output, h_n = self.rnn(x)
        # print(output) # prints all the hidden outputs
        h_n = h_n.squeeze(0)
        # print(h_n) # prints only the last hidden output
        
        final = self.fc(h_n)
        return final
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.squeeze()
        # x is a list of letter indexes from start to end
        x = self.embedding(x) # shape word_length, 16
        # print(x)
        output, h_n = self.rnn(x)
        # print(output) # prints all the hidden outputs
        h_n = h_n.squeeze()
        # print(h_n) # prints only the last hidden output
        
        final = self.fc(h_n)
        return torch.softmax(final, dim=-1)
        # return final


def to_tensor(name: str) -> torch.Tensor:
    name = name.lower()
    lst: list[int] = []
    for c in name:
        n: int = ord(c) - ord('a') + 1
        if n >= 1 and n <= 26:
            lst.append(n)
        else:
            lst.append(0)
    return pad(torch.tensor(lst))

def pad(name: torch.Tensor, max_len = 10) -> torch.Tensor:
    name = torch.cat([name, torch.tensor([0 for i in range(max_len - len(name))])])
    name = name.int()
    return name[:max_len] # ensures name is not too long


if __name__ == "__main__":
    m = Model()

    # print(m(torch.tensor([0])))
    # print(m(torch.tensor([2, 3, 5, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])))

    n1 = to_tensor("justin")
    n2 = to_tensor("simon")
    n3 = to_tensor("ursulakia")
    # print(n1)
    print(m(torch.stack([n1, n2, n3])))
    k = n1.unsqueeze(0)
    print(k)
    print(m(k))
    print(m.predict(n1))


