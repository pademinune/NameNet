
import torch
from to_vector import to_tensor

from neural_net import Model

model = Model()

version = "v1"

model.load_state_dict(torch.load(f"models/{version}/{version}.model"))
model.eval()

# print(model(to_tensor("archie")))


while True:
    inp: str = input("Enter a name: ")
    out: torch.Tensor = model(to_tensor(inp))
    print(f"Male: {out[0].item():.3f} | Female: {out[1].item():.3f}")
