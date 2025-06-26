
import torch
import to_vector

from neural_net import Model
from name_net_2 import NameNet2

model = NameNet2()

version = "v2"

model.load_state_dict(torch.load(f"models/{version}/{version}.model"))
# model.load_state_dict(torch.load("trained.model"))
model.eval()

# print(model(to_tensor("archie")))


while True:
    inp: str = input("Enter a name: ")
    # k = to_vector.index_tensor(inp)
    out: torch.Tensor = model.predict(to_vector.index_tensor(inp).unsqueeze(0))
    out = out.squeeze()
    print(f"Male: {out[0].item():.3f} | Female: {out[1].item():.3f}")
