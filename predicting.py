
import torch


import architectures.r1 as r1

import architectures.v1 as v1
import architectures.v2 as v2
import architectures.v3 as v3


model = v3.Model()

version = "v3"

# model.load_state_dict(torch.load(f"models/{version}/{version}.3.small.model"))
model.load_state_dict(torch.load(f"models/{version}/{version}.0.model"))


model.eval()


while True:
    inp: str = input("Enter a name: ")
    out = model.predict_name(inp)
    print(f"Male: {out[0]:.3f} | Female: {out[1]:.3f}")
