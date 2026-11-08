import torch
import torch.nn as nn

# 1. Исходные данные

apartments = [
    [30, 1],
    [40, 1],
    [50, 2],
    [60, 2],
    [75, 3],
    [90, 3],
    [110, 4],
    [120, 4]
]

prices = [4.2, 5.1, 6.5, 7.4, 9.3, 10.8, 13.5, 14.7]


# 2. Нормализация квартир

normalized_apartments = []

for apartment in apartments:
    area = apartment[0] / 120
    rooms = apartment[1] / 4
    normalized_apartments.append([area, rooms])

x = torch.tensor(normalized_apartments).float()
y = torch.tensor(prices).float()


# 3. Разделяем данные на Train и Test

x_train = x[:6]
y_train = y[:6]

x_test = x[6:]
y_test = y[6:]

print("Train:", x_train.shape, y_train.shape)
print("Test:", x_test.shape, y_test.shape)


# 4. Создаём ОДНУ нейросеть

model = nn.Linear(2, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# 5. Обучаем только на Train

epochs = 500

for epoch in range(epochs):
    optimizer.zero_grad()

    prediction = model(x_train).squeeze(1)

    loss = criterion(prediction, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print("Эпоха", epoch + 1, "Ошибка", loss.item())


# 6. Проверяем обученную модель

with torch.no_grad():

    train_prediction = model(x_train).squeeze(1)
    test_prediction = model(x_test).squeeze(1)

    train_loss = criterion(train_prediction, y_train).item()
    test_loss = criterion(test_prediction, y_test).item()


# 7. Вычисляем RMSE

train_rmse = train_loss ** 0.5
test_rmse = test_loss ** 0.5

print("Train MSE:", round(train_loss, 4))
print("Test MSE:", round(test_loss, 4))

print("Train RMSE:", round(train_rmse, 4))
print("Test RMSE:", round(test_rmse, 4))


# 8. Смотрим предсказания тестовых квартир

total_error = 0
mae = 0
for i in range(len(y_test)):
    errors = abs(y_test[i].item() - test_prediction[i].item())
    total_error += errors
    print(
    "Настоящая цена:", round(y_test[i].item(), 2),
    "Предсказание:", round(test_prediction[i].item(), 2),
    'Абсолютная ошибка', errors,

)
mae_sum = total_error / len(y_test)
print('Средняя абсолютная ошибка', mae_sum)
