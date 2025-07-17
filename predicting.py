
import torch
import to_vector

import architectures.r1 as r1


import architectures.v1 as v1
import architectures.v2 as v2



model = v2.NameNet2()

version = "v2"

# model.load_state_dict(torch.load(f"models/{version}/{version}.3.small.model"))
model.load_state_dict(torch.load(f"models/{version}/{version}.3.small.model"))
# model.load_state_dict(torch.load("trained.model"))
model.eval()

# print(model(to_tensor("archie")))


while True:
    inp: str = input("Enter a name: ")
    # k = to_vector.index_tensor(inp)
    out: torch.Tensor = model.predict(v2.to_tensor(inp).unsqueeze(0))
    out = out.squeeze()
    # out: torch.Tensor = model(r1.to_tensor(inp))
    # out = torch.softmax(out, dim=-1)
    print(f"Male: {out[0].item():.3f} | Female: {out[1].item():.3f}")
