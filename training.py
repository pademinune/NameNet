import torch
import torch.nn as nn
import torch.optim as optim

from dataset import dataloader

import architectures.v1 as v1
import architectures.v2 as v2
import architectures.r1 as r1

import time

m: r1.Model = r1.Model()
# for features, labels in dataloader:
#     print(labels)

optimizer = optim.Adam(m.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# import to_vector
# print(m())

# for features,labels in dataloader:
#     print(features.shape)
#     print(features[0])

epochs: int = 10
loss_interval: int = max(1, epochs // 10)

print(f"Training started on {epochs} epochs.")

for epoch in range(epochs):
    
    if epoch % loss_interval == 0:
        start = time.time()

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
        end = time.time()
        duration = end - start
        print(f"Epoch {epoch} loss: {epoch_loss/num_batches:.5f}\t\t\t{duration:.3f}s")

torch.save(m.state_dict(), "trained.model")
print("TRAINING FINISHED")


