import torch
import torch.nn as nn

class XORNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        hidden = torch.relu(self.hidden(x))
        output = self.output(hidden)
        return output

model = XORNetwork()
saved_state = torch.load('xor_model.pth', weights_only = True)
model.load_state_dict(saved_state)
model.eval()

inputs = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

with torch.no_grad():
    predictions = model(inputs)
print(predictions)

print("Старая модель:", predictions)
