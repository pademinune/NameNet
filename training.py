import torch
import torch.nn as nn
import torch.optim as optim

from dataset import dataloader

from neural_net import Model
from name_net_2 import NameNet2

m: NameNet2 = NameNet2()
# for features, labels in dataloader:
#     print(labels)

optimizer = optim.Adam(m.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# import to_vector
# print(m())

# for features,labels in dataloader:
#     print(features.shape)
#     print(features[0])

epochs: int = 1000
loss_interval: int = epochs // 10

for epoch in range(epochs):
    epoch_loss: float = 0
    num_batches: int = 0

    for features, labels in dataloader:
        out: torch.Tensor = m(features)

        loss: torch.Tensor = loss_fn(out, labels)
        epoch_loss += loss.item()
        num_batches += 1

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if epoch % loss_interval == 0:
        print(f"Epoch {epoch} loss: {epoch_loss/num_batches:.5f}")

torch.save(m.state_dict(), "trained.model")
print("TRAINING FINISHED")


