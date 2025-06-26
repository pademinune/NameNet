import torch
import torch.nn as nn
import torch.optim as optim

from dataset import dataloader

from neural_net import Model
    

m: Model = Model()


optimizer = optim.Adam(m.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs: int = 10000
loss_interval: int = epochs // 10

for epoch in range(epochs):
    for features, labels in dataloader:
        out: torch.Tensor = m(features)

        loss: torch.Tensor = loss_fn(out, labels)
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if epoch % loss_interval == 0:
        print(f"Epoch {epoch} loss: {loss.item():.5f}")

torch.save(m.state_dict(), "trained.model")
print("TRAINING FINISHED")


