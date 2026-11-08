import torch
import torch.nn as nn

class XORNetwork (nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2,8)
        self.output = nn.Linear(8,1)

    def forward(self , x):
        hidden = torch.relu(self.hidden(x))
        output = self.output(hidden)
        return output
model = XORNetwork()
print(model)

inputs = torch.tensor([1.0, 0.0])
prediction = model(inputs)
print(prediction)
print(prediction.shape)

training_inputs = torch.tensor([
    [0.0,0.0],
    [0.0,1.0],
    [1.0,0.0],
    [1.0,1.0]
])

predictions = model(training_inputs)
print(predictions.shape)

targets = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])
print(targets.shape)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

epochs = 2000

for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = model(training_inputs)

    loss = loss_fn(prediction, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(
        "Эпохи:", epoch + 1,
        "Номер ошибки:", loss.item()
        )
model.eval()
with torch.no_grad():
    predictions = model(training_inputs)
print("Предсказания:", predictions)
print("Правильные ответы:", targets)

model_state =model.state_dict()
torch.save(model_state, "xor_model.pth")

print("Модель сохранена!")

new_model = XORNetwork()

saved_state = torch.load("xor_model.pth", weights_only = True)

new_model.load_state_dict(saved_state)

new_model.eval()

with torch.no_grad():
    new_prediction = new_model(training_inputs)
print(new_prediction)

print("Старая модель:", predictions)
print("Загруженная модель:", new_prediction)

result = torch.allclose(predictions, new_prediction, atol = 1e-6)
print(result)
