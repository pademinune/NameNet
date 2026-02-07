
import torch


import models.r1 as r1

import models.v1 as v1
import models.v2 as v2
import models.v3 as v3
import models.t1 as t1

model: r1.Model = r1.Model() # model architecture


version = "r1"

# model.load_state_dict(torch.load(f"trained_models/{version}/{version}.3.small.model"))
model.load_state_dict(torch.load(f"trained_models/{version}/{version}.2.1.model"))
# model.load_state_dict(torch.load(f"trained_models/{version}/{version}.model"))



model.eval()

print(f"Loading model {model.name}...")

while True:
    inp: str = input("Enter a name: ")
    out = model.predict_name(inp)
    print(f"Male: {out[0]:.3f} | Female: {out[1]:.3f}")

