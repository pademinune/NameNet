
import torch
from to_vector import to_tensor

from neural_net import Model

model = Model()

version = "v1"

model.load_state_dict(torch.load(f"{version}.model"))
model.eval()

print(model(to_tensor("archie")))

