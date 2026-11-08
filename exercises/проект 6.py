"""Проект №6. Небольшая нейросеть для задачи XOR на чистом Python.

Сеть: 2 входа -> 3 скрытых нейрона -> 1 выходной нейрон.
Обучение: градиентный спуск с численным приближением градиентов.
Без сторонних библиотек.
"""


# 1. ОСНОВНЫЕ ВЫЧИСЛЕНИЯ НЕЙРОСЕТИ

def weighted_sum(inputs, weights):
    """Складываем произведения каждого входа на соответствующий вес."""
    total = 0
    for i in range(len(inputs)):
        total += inputs[i] * weights[i]
    return total


def relu(x):
    """Если число отрицательное, возвращаем ноль."""
    return x if x > 0 else 0


def neuron(inputs, weights, bias):
    """Один нейрон: взвешенная сумма, смещение и ReLU."""
    total = weighted_sum(inputs, weights)
    total += bias
    output = relu(total)
    return output


def layer(inputs, weights, biases):
    """Запускаем все нейроны одного слоя и собираем их ответы."""
    outputs = []
    for i in range(len(biases)):
        output = neuron(inputs, weights[i], biases[i])
        outputs.append(output)
    return outputs


def network(inputs, hidden_weights, hidden_biases, output_weights, output_bias):
    """Первый слой выдаёт три числа, выходной нейрон — одно."""
    hidden = layer(inputs, hidden_weights, hidden_biases)
    output = neuron(hidden, output_weights, output_bias)
    return output


# 2. ОШИБКА НЕЙРОСЕТИ

def squared_error(prediction, target):
    """Квадрат разницы между предсказанием и правильным ответом."""
    return (target - prediction) ** 2


def dataset_loss(training_data, hidden_weights, hidden_biases,
                 output_weights, output_bias):
    """Средняя квадратичная ошибка по всем четырём примерам XOR."""
    total_error = 0
    for inputs, target in training_data:
        prediction = network(
            inputs, hidden_weights, hidden_biases,
            output_weights, output_bias
        )
        total_error += squared_error(prediction, target)
    return total_error / len(training_data)


# 3. ГРАДИЕНТЫ ВЕСОВ
# На время слегка меняем вес, сравниваем ошибки и возвращаем вес обратно.

def weight_gradient(index, training_data, hidden_weights,
                    hidden_biases, output_weights, output_bias, step):
    """Приближённый градиент одного веса выходного нейрона."""
    old_loss = dataset_loss(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias
    )
    output_weights[index] += step
    new_loss = dataset_loss(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias
    )
    output_weights[index] -= step
    return (new_loss - old_loss) / step


def output_gradients(training_data, hidden_weights, hidden_biases,
                     output_weights, output_bias, step):
    """Градиенты всех трёх весов выходного нейрона."""
    gradients = []
    for index in range(len(output_weights)):
        gradient = weight_gradient(
            index, training_data, hidden_weights,
            hidden_biases, output_weights, output_bias, step
        )
        gradients.append(gradient)
    return gradients


def hidden_weight_gradient(neuron_index, weight_index, training_data,
                           hidden_weights, hidden_biases, output_weights,
                           output_bias, step):
    """Приближённый градиент одного веса скрытого нейрона."""
    old_loss = dataset_loss(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias
    )
    hidden_weights[neuron_index][weight_index] += step
    new_loss = dataset_loss(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias
    )
    hidden_weights[neuron_index][weight_index] -= step
    return (new_loss - old_loss) / step


def hidden_gradients(training_data, hidden_weights, hidden_biases,
                     output_weights, output_bias, step):
    """Градиенты шести весов скрытого слоя (список списков)."""
    gradients = []
    for i in range(len(hidden_weights)):
        neuron_gradients = []
        for j in range(len(hidden_weights[i])):
            gradient = hidden_weight_gradient(
                i, j, training_data, hidden_weights,
                hidden_biases, output_weights, output_bias, step
            )
            neuron_gradients.append(gradient)
        gradients.append(neuron_gradients)
    return gradients


# 4. ГРАДИЕНТЫ СМЕЩЕНИЙ (BIAS)

def output_bias_gradient(training_data, hidden_weights, hidden_biases,
                         output_weights, output_bias, step):
    """Приближённый градиент единственного выходного bias."""
    old_loss = dataset_loss(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias
    )
    new_loss = dataset_loss(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias + step
    )
    return (new_loss - old_loss) / step


def hidden_bias_gradient(index, training_data, hidden_weights,
                         hidden_biases, output_weights, output_bias, step):
    """Приближённый градиент одного bias скрытого слоя."""
    old_loss = dataset_loss(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias
    )
    hidden_biases[index] += step
    new_loss = dataset_loss(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias
    )
    hidden_biases[index] -= step
    return (new_loss - old_loss) / step


def hidden_bias_gradients(training_data, hidden_weights, hidden_biases,
                          output_weights, output_bias, step):
    """Градиенты всех трёх bias скрытого слоя."""
    gradients = []
    for i in range(len(hidden_biases)):
        gradient = hidden_bias_gradient(
            i, training_data, hidden_weights,
            hidden_biases, output_weights, output_bias, step
        )
        gradients.append(gradient)
    return gradients


# 5. ОБНОВЛЕНИЕ ПАРАМЕТРОВ
# Вычитаем градиент, умноженный на скорость обучения.

def update_output_weights(output_weights, gradients, learning_rate):
    """Обновляем веса выходного нейрона."""
    for i in range(len(output_weights)):
        output_weights[i] -= learning_rate * gradients[i]
    return output_weights


def update_hidden_weights(hidden_weights, gradients, learning_rate):
    """Обновляем все веса скрытого слоя."""
    for i in range(len(hidden_weights)):
        for j in range(len(hidden_weights[i])):
            hidden_weights[i][j] -= learning_rate * gradients[i][j]
    return hidden_weights


def update_biases(hidden_biases, output_bias, hidden_bias_grads,
                  output_bias_grad, learning_rate):
    """Обновляем скрытые смещения и одно выходное смещение."""
    for i in range(len(hidden_biases)):
        hidden_biases[i] -= learning_rate * hidden_bias_grads[i]
    output_bias -= learning_rate * output_bias_grad
    return hidden_biases, output_bias


# 6. ОБУЧЕНИЕ

def train_full_step(training_data, hidden_weights, hidden_biases,
                    output_weights, output_bias, step, learning_rate):
    """Один шаг: все градиенты -> обновление всех 13 параметров -> ошибка."""
    # Сначала вычисляем ВСЕ градиенты при старых параметрах.
    output_grads = output_gradients(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias, step
    )
    hidden_grads = hidden_gradients(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias, step
    )
    output_bias_grad = output_bias_gradient(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias, step
    )
    hidden_bias_grads = hidden_bias_gradients(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias, step
    )

    # Только теперь обновляем веса и смещения.
    output_weights = update_output_weights(
        output_weights, output_grads, learning_rate
    )
    hidden_weights = update_hidden_weights(
        hidden_weights, hidden_grads, learning_rate
    )
    hidden_biases, output_bias = update_biases(
        hidden_biases, output_bias,
        hidden_bias_grads, output_bias_grad, learning_rate
    )

    loss = dataset_loss(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias
    )
    return hidden_weights, hidden_biases, output_weights, output_bias, loss


def train_full(training_data, hidden_weights, hidden_biases,
               output_weights, output_bias, step, learning_rate, epochs):
    """Повторяем полный шаг обучения указанное число эпох."""
    for epoch in range(epochs):
        hidden_weights, hidden_biases, output_weights, output_bias, loss = (
            train_full_step(
                training_data, hidden_weights, hidden_biases,
                output_weights, output_bias, step, learning_rate
            )
        )
    return hidden_weights, hidden_biases, output_weights, output_bias, loss


# 7. ЗАПУСК И ПРОВЕРКА XOR

def main():
    # В XOR правильный ответ равен 1, если входы различаются.
    training_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    # Начальные параметры сети: 2 входа -> 3 нейрона -> 1 нейрон.
    hidden_weights = [
        [0.5, -0.2],
        [0.3, 0.8],
        [-0.4, 0.1],
    ]
    hidden_biases = [0.1, 0.0, 0.2]
    output_weights = [0.4, -0.2, 0.7]
    output_bias = 0.1

    step = 0.01          # Насколько меняем параметр при оценке градиента.
    learning_rate = 0.1  # Насколько сильно обновляем параметр.
    epochs = 100        # Сколько раз повторяем обучение.

    old_loss = dataset_loss(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias
    )

    hidden_weights, hidden_biases, output_weights, output_bias, loss = train_full(
        training_data, hidden_weights, hidden_biases,
        output_weights, output_bias, step, learning_rate, epochs
    )

    print("Ошибка до обучения:", old_loss)
    print("Ошибка после обучения:", loss)
    print()

    correct = 0
    for inputs, target in training_data:
        prediction = network(
            inputs, hidden_weights, hidden_biases,
            output_weights, output_bias
        )
        predicted_class = 1 if prediction >= 0.5 else 0
        if predicted_class == target:
            correct += 1

        print(
            "Вход:", inputs,
            "| Правильный:", target,
            "| Предсказание:", round(prediction, 4),
            "| Класс:", predicted_class,
        )

    accuracy = correct / len(training_data)
    print("Точность:", accuracy * 100, "%")


if __name__ == "__main__":
    main()
