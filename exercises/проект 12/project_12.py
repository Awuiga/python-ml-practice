import torch
import torch.nn as nn

texts = [
    'Привет как дела',
    'Вы выиграли миллион заберите приз',
    'Давай встретимся завтра',
    'Получите кредит без проверки прямо сейчас',
    'Купи пожалуйста молоко',
    'Бесплатный подарок перейдите по ссылке',
    'Во сколько начинаются занятия',
    'Срочно получите денежный бонус'
]

labels = [0,1,0,1,0,1,0,1]

print(texts[1])
print(labels[1])
print(len(texts) == len(labels))

tokenized_texts = []

for words in tuple(texts):
    tokenized_texts.append(words.lower().split())
print(tokenized_texts)

vocab = {}

for words in tuple(tokenized_texts):
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
print(vocab)

bow = [0] * len(vocab)
print(bow)
print(len(bow))

for word in tokenized_texts[0]:
    index = vocab[word]
    bow[index] = 1
print(bow)

all_bows = []

for words in tokenized_texts:
    bow = [0] * len(vocab)
    for word in words:
        index = vocab[word]
        bow[index] = 1
    all_bows.append(bow)
print(all_bows)

x = torch.tensor(all_bows).float()
print(x.shape)
print(x.dtype)

y = torch.tensor(labels)
print(y.shape)
print(y.dtype)


layer = nn.Linear(32,2)
scores = layer(x)
print(scores.shape)
criterion = nn.CrossEntropyLoss()
loss = criterion(scores, y)
optimizer = torch.optim.SGD(layer.parameters(), lr = 0.1)

epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()
    scores = layer(x)
    loss = criterion(scores, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print('Эпоха ', epoch + 1,
              'Ошибка', loss.item())

with torch.no_grad():
    scores = layer(x)
    prediction = scores.argmax(dim = 1)
print(prediction, y)

correct_count = (prediction == y).sum().item()
total = len(y)
accuracy = round((correct_count / total) * 100, 2)
print(accuracy)

for i in range(len(texts)):
    print(texts[i] , '- Спам' if prediction[i].item() == 1 else '- Обычное')

test_texts = [
    'Получите подарок',
    'Давай завтра',
    'Срочно выиграли приз'
]
errors = total - correct_count

test_tokens = []
for words in tuple(test_texts):
    test_tokens.append(words.lower().split())
print(test_tokens)

test_bows = []
for words in test_tokens:
    bows = [0] * len(vocab)
    for word in words:
        index = vocab[word]
        bows[index] = 1
    test_bows.append(bows)
print(test_bows)

test_x = torch.tensor(test_bows).float()
print(test_x.shape)
print(test_x.dtype)

with torch.no_grad():
    scores = layer(test_x)
    test_predictions = scores.argmax(dim = 1)

for i in range(len(test_texts)):
    print(test_texts[i], '- Спам' if test_predictions[i].item() == 1 else ' - Обычное')

def predict_spam(text):
    new_text = tuple(text.lower().split())
    bow = [0] * len(vocab)
    for word in new_text:
        if word in vocab:
            index = vocab[word]
            bow[index] = 1
    input_tensor = torch.tensor([bow]).float()

    with torch.no_grad():
        scores = layer(input_tensor)
        more_predicion = scores.argmax(dim = 1)
    result = more_predicion.item()
    return 'Спам' if result == 1 else 'Обычное'
print(predict_spam("Привет кот"))
print(predict_spam("Срочно получите приз"))
