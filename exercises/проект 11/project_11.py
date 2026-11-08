import torch
import torch.nn as nn
text = "кот ест рыбу кот пьет молоко пес ест мясо пес пьет воду"
words = text.split()
def word_to_id(words):
    vocab = {}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab
vocab = word_to_id(words)
print(vocab)

token_ids = [vocab[word] for word in words]
print(token_ids)

inputs = []
targets = []

for i in range(0, len(token_ids), 3):
    triple = token_ids[i:i +3]
    inputs.append(triple[:2])
    targets.append(triple[2])

ids = torch.tensor(inputs)
targets = torch.tensor(targets)

embedding = nn.Embedding(len(vocab), 3)
vectors = embedding(ids)
contexts = vectors.mean(dim = 1)

print(vectors.shape)
print(contexts.shape)

layer = nn.Linear(3, len(vocab))

scores = layer(contexts)
print(scores.shape)

criterion = nn.CrossEntropyLoss()
loss = criterion(scores, targets)
print(loss.shape)

params = (list(layer.parameters()) + list(embedding.parameters()))
optimizer = torch.optim.SGD(params, lr = 0.1)
optimizer.zero_grad()
print(loss.item())

epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()
    vectors = embedding(ids)
    contexts = vectors.mean(dim = 1)
    scores = layer(contexts)
    loss = criterion(scores, targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print('Эпоха ',epoch + 1,
            'номер ошибки ', loss.item())

with torch.no_grad():
    vectors = embedding(ids)
    contexts = vectors.mean(dim = 1)
    scores = layer(contexts)
    predictions = scores.argmax(dim = 1)
print(predictions)
print(targets)

id_vocab = {}
for word, word_id in vocab.items():
    id_vocab[word_id] = word
print(id_vocab)

for predicted_id in predictions:
    number = predicted_id.item()
    word = id_vocab[number]
    print(word)


correct_count = (predictions == targets).sum().item()
total = len(targets)
accuracy = round((correct_count / total)  * 100,2)
print(accuracy)

user_text = input("Введите два слова: ")
user_words = user_text.split()
if len(user_words) != 2:
    print( "Вы ввели неправильное количество слов")
else:
    print (user_words)
    user_ids = []
    for word in user_words:
        if word in vocab:
            user_ids.append(vocab[word])
        else:
            print("Неизвестное слово: " , word)
    if len(user_ids) == 2:
        with torch.no_grad():
            user_tensor = torch.tensor([user_ids])
            user_vectors = embedding(user_tensor)
            user_context = user_vectors.mean(dim = 1)
            user_scores = layer(user_context)
            predicted_id = user_scores.argmax(dim = 1)
            number = predicted_id.item()
            predicted_word = id_vocab[number]
            print(predicted_word)
