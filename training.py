import torch
import torch.nn as nn
import torch.optim as optim

from name_dataset import good

from to_vector import to_tensor

from neural_net import Model
    

m: Model = Model()


# features, labels = read_names("names.txt")
features, labels = good

optimizer = optim.Adam(m.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs: int = 10000
loss_interval: int = epochs // 10

for epoch in range(epochs):
    out: torch.Tensor = m(features)

    loss: torch.Tensor = loss_fn(out, labels)

    if epoch % loss_interval == 0:
        print(f"Epoch {epoch} loss: {loss.item():.5f}")
    
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

torch.save(m.state_dict(), "model.pth")

# print([torch.argmax(t).item() for t in m(features)])

print(m.forward(to_tensor("justin")))
print(m.forward(to_tensor("raluca")))

