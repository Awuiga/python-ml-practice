inputs = [1, 0, 1]
weights = [0.5, 0.2, -0.3]
bias = 0.1

def weighted_sum(inputs, weights):
    total = 0
    for i in range(len(inputs)):
        total += weights[i] * inputs[i]
    return total

print(weighted_sum(inputs, weights))


def relu(x):
    if x > 0:
        return x
    else:
        return 0

def neuron(inputs, weights, bias):
    total = weighted_sum(inputs, weights)
    total += bias
    output = relu(total)
    return output

def squared_error(prediction, target):
    error = (prediction - target) ** 2
    return error
print(squared_error(0.3, 1.0))

def get_difference(prediction, target):
    difference = target - prediction
    return difference
print(get_difference(0.3, 1.0))  # 0.7
print(get_difference(1.4, 1.0))  # -0.4

def update_weights(inputs, weights, difference, learning_rate):
    new_weights = []

    for i in range(len(weights)):
        correction = learning_rate * difference * inputs[i]
        new_weight = weights[i] + correction

        new_weights.append(new_weight)

    return new_weights

difference = 0.7
learning_rate = 0.1

print(update_weights(inputs, weights, difference, learning_rate))

def update_bias(bias, difference, learning_rate):
    correction = learning_rate * difference
    new_bias = bias + correction
    return new_bias

bias = 0.1
difference = 0.7
learning_rate = 0.1

print(update_bias(bias, difference, learning_rate))

def train_step(inputs, weights, bias, target, learning_rate):
    predict = neuron(inputs, weights, bias)
    error = squared_error(predict, target)
    difference = get_difference(predict, target)
    new_weights = update_weights(inputs, weights, difference, learning_rate)
    new_bias = update_bias(bias, difference, learning_rate)

    return new_weights, new_bias, predict, error

inputs = [1, 0, 1]
weights = [0.5, 0.2, -0.3]
bias = 0.1

target = 1.0
learning_rate = 0.1

result = train_step(
    inputs,
    weights,
    bias,
    target,
    learning_rate
)

print(result)

def train(inputs, weights, bias, target, learning_rate, epochs):
    for _ in range(epochs):
        new_weights, new_bias, predict, error = train_step(inputs, weights, bias, target, learning_rate)

        weights = new_weights
        bias = new_bias

    return weights, bias, predict, error

result = train(
    inputs=[1, 0, 1],
    weights=[0.5, 0.2, -0.3],
    bias=0.1,
    target=1.0,
    learning_rate=0.1,
    epochs=10
)

print(result)

def train(inputs, weights, bias, target, learning_rate, epochs):
    for epoch in range(epochs):
        new_weights, new_bias, predict, error = train_step(
            inputs,
            weights,
            bias,
            target,
            learning_rate
        )

        weights = new_weights
        bias = new_bias
        print(
            'epoch =', epoch + 1,
            'prediction =', predict,
            'error =', error
        )
    return weights, bias, predict, error

result = train(
    inputs=[1, 0, 1],
    weights=[0.5, 0.2, -0.3],
    bias=0.1,
    target=1.0,
    learning_rate=0.1,
    epochs=10
)

print("Итог:", result)

training_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

def test_neuron(training_data, weights, bias):
    result = []

    for inputs, target in training_data:
        prediction = neuron(inputs, weights, bias)

        result.append((inputs, target, prediction))
    return result

def train_dataset(training_data, weights, bias, learning_rate, epochs):
    for epoch in range(epochs):
        for inputs, target in training_data:
            new_weights, new_bias, prediction, error = train_step(
                inputs,
                weights,
                bias,
                target,
                learning_rate
            )

            weights = new_weights
            bias = new_bias

    return weights, bias

training_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

weights = [0.0, 0.0]
bias = 0.0
learning_rate = 0.1
epochs = 50

weights, bias = train_dataset(
    training_data,
    weights,
    bias,
    learning_rate,
    epochs
)

results = test_neuron(
    training_data,
    weights,
    bias
)

print("weights =", weights)
print("bias =", bias)

for inputs, target, prediction in results:
    print(
        "inputs =", inputs,
        "target =", target,
        "prediction =", prediction
    )

def to_class(prediction):
    return (1 if prediction >= 0.5 else 0)

def accuracy(training_data, weights, bias):
    correction = 0
    for inputs, target in training_data:
        prediction = neuron(inputs, weights, bias)
        predict_class = to_class(prediction)

        if predict_class == target:
            correction += 1

    result = correction / len(training_data)

    return result
weights, bias = train_dataset(
    training_data,
    [0.0, 0.0],
    0.0,
    0.1,
    50
)

print("weights:", weights)
print("bias:", bias)
print("accuracy:", accuracy(training_data, weights, bias))
