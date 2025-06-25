import torch


def to_tensor(name: str) -> torch.Tensor:
    tensor: torch.Tensor = torch.zeros(26)

    for char in name:
        char = char.lower()
        index: int = ord(char) - ord('a')

        tensor[index] += 1

    return tensor

def label_to_tensor(label: str) -> torch.Tensor:
    if label == "Male":
        return torch.tensor([1.0, 0])
    else:
        return torch.tensor([0, 1.0])

if __name__ == "__main__":
    print(to_tensor("Phil"))
