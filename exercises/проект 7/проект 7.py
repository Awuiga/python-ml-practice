import torch
import torch.nn as nn


inputs = torch.tensor([1.0, 0.0])

layer = nn.Linear(2, 1)

print(layer(inputs))
print(layer.weight)
print(layer.bias)

inputs = torch.tensor([1.0, 0.0])
weights = torch.tensor([0.5, -0.2])


def weighted_sum_torch(inputs, weights):
    multiply = inputs * weights
    return torch.sum(multiply)

def relu_torch(x):
    return torch.relu(x)

def neuron_torch(inputs, weights, bias):
    total = weighted_sum_torch(inputs, weights) + bias
    return relu_torch(total)

hidden_layer = nn.Linear(2,3)
output_layer = nn.Linear(3,1)

hidden = relu_torch(hidden_layer(inputs))

output = output_layer(hidden)

print("Скрытый слой:", hidden)
print("Выход сети:", output)

x = torch.tensor(3.0, requires_grad = True)

y = x ** 2

y.backward()

print(x.grad)

inputs = torch.tensor(3.0)
weight = torch.tensor(2.0, requires_grad=True)
target = torch.tensor(10.0)

prediction = inputs * weight

loss = (prediction - target) ** 2

loss.backward()

print("Предсказание:", prediction)
print("Ошибка:", loss)
print("Градиент веса:", weight.grad)

learning_rate = 0.1

new_weight = weight - learning_rate * weight.grad

print("Новый вес:", new_weight)

new_prediction = inputs * new_weight

new_loss = (new_prediction - target) ** 2

print("Старая ошибка:", loss)
print("Новая ошибка:", new_loss)
print("Новое предсказание:", new_prediction)

inputs = torch.tensor(3.0)
weight = torch.tensor(2.0, requires_grad=True)
target = torch.tensor(10.0)

learning_rate = 0.1

prediction = inputs * weight
loss = (prediction - target) ** 2

loss.backward()

with torch.no_grad():
    weight -= learning_rate * weight.grad

print("Обновлённый вес:", weight)

inputs = torch.tensor(3.0)
weight = torch.tensor(2.0, requires_grad=True)
target = torch.tensor(10.0)

prediction = inputs * weight
loss = (prediction - target) ** 2

loss.backward()

print("Градиент до очистки:", weight.grad)

weight.grad.zero_()

print("Градиент после очистки:", weight.grad)

learning_rate = 0.05
epochs = 10
for epoch in range(epochs):
    prediction = inputs * weight
    loss = (prediction - target) ** 2
    loss.backward()

    with torch.no_grad():
        weight -= learning_rate * weight.grad
    weight.grad.zero_()

    print(
        "Эпоха:", epoch + 1,
        "Ошибка до обновления:", loss.item()
    )

inputs = torch.tensor(3.0)
weight = torch.tensor(2.0, requires_grad=True)
target = torch.tensor(10.0)

optimizer = torch.optim.SGD([weight], lr=0.05)

epochs = 10

for epoch in range(epochs):
    # 1. Получаем предсказание
    predict = inputs * weight

    # 2. Считаем ошибку
    loss = (predict - target) ** 2

    # 3. Вычисляем градиент
    loss.backward()

    # 4. Обновляем вес
    optimizer.step()

    # 5. Очищаем градиент
    optimizer.zero_grad()

    print(
        "Эпоха:", epoch + 1,
        "Ошибка:", loss.item()
    )

print("Итоговый вес:", weight.item())

hidden_layer = nn.Linear(2, 8)
output_layer = nn.Linear(8, 1)
training_inputs = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

targets = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

def forward_torch(inputs, hidden_layer, output_layer):
    hidden = torch.relu(hidden_layer(inputs))

    output = output_layer(hidden)

    return output
predictions = forward_torch(
    training_inputs,
    hidden_layer,
    output_layer
)

print(predictions)
print(predictions.shape)

loss_fn = nn.MSELoss()

prediction = forward_torch(training_inputs, hidden_layer, output_layer)

loss = loss_fn(prediction, targets)

print("Ошибка нейросети:", loss.item())

hidden_params = list(hidden_layer.parameters())
output_params = list(output_layer.parameters())

params = hidden_params + output_params

optimizer = torch.optim.Adam(params, lr = 0.05)

epochs = 2000

for epoch in range(epochs):
    prediction = forward_torch(training_inputs, hidden_layer, output_layer)
    loss = loss_fn(prediction, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(
            "Эпоха:", epoch + 1,
            "Ошибка:", loss.item()
        )

with torch.no_grad():
    predictions = forward_torch(
        training_inputs,
        hidden_layer,
        output_layer
    )




# Превращаем предсказания в 0 и 1
classes = (predictions >= 0.5).float()

# Сравниваем с правильными ответами,
correct = (classes == targets).float()
# преобразуем True/False в числа и находим среднее
accuracy = correct.mean()

# Выводим точность в процентах
print("Точность:", accuracy.item() * 100, "%")



with torch.no_grad():
    predictions = forward_torch(
        training_inputs,
        hidden_layer,
        output_layer
    )

print("Предсказания:", predictions)
print("Правильные ответы:", targets)
