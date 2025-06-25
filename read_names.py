
from to_vector import to_tensor
import torch


def read_names(file_name: str ="names.txt") -> tuple[torch.Tensor, torch.Tensor]:
    tensors = []
    labels = []

    with open(file_name, 'r') as file:
        for line in file.readlines():
            name, label = line.split()
            tensors.append(to_tensor(name))

            if label == "Male":
                labels.append(torch.tensor([1.0, 0]))
            else:
                labels.append(torch.tensor([0, 1.0]))

    return torch.stack(tensors), torch.stack(labels)

