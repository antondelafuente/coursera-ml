import numpy as np
import torch
import torch.nn as nn

data = np.loadtxt('ex1data1.txt', delimiter=',')
x = torch.tensor(data[:,  0]).float().reshape(-1, 1)
y = torch.tensor(data[:, -1]).float().reshape(-1, 1)

model = nn.Linear(x.shape[1], 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(5000):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

theta = torch.cat((model.bias[0].unsqueeze(0), model.weight[0])).detach().numpy()
print(theta)
