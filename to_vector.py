import torch


def to_tensor(name: str) -> torch.Tensor:
    tensor: torch.Tensor = torch.zeros(26)

    for char in name:
        char = char.lower()
        index: int = ord(char) - ord('a')

        tensor[index] += 1

    return tensor

def index_tensor(name: str) -> torch.Tensor:
    """Returns a right-aligned tensor of the characters"""
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


def label_to_tensor(label: str) -> torch.Tensor:
    if label == "Male":
        return torch.tensor([1.0, 0])
    else:
        return torch.tensor([0, 1.0])

if __name__ == "__main__":
    print(to_tensor("Phil"))
