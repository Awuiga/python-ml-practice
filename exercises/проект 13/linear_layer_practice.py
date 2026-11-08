import torch
import torch.nn as nn

x = torch.tensor([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])

y = torch.tensor([
    [3.],
    [7.],
    [11.]
])

model = nn.Linear(2, 1)
criterion = nn.MSELoss()

# Твой код
prediction = model(x)
loss = criterion(prediction, y)

print(prediction.shape)
print(loss.item())
