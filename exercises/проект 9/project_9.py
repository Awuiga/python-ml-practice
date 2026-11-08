import torch
import torch.nn as nn

text = ("кот ест рыбу кот спит")

words = text.split()

print(words, len(words))

word_to_id = {}

for word in words:
    if word not in word_to_id:
        word_to_id[word] = len(word_to_id)
print(word_to_id)

token_ids = []

for word in words:
    token_ids.append(word_to_id[word])
print(token_ids)
token_tensor = torch.tensor(token_ids, dtype = torch.long)
print(token_tensor)
print(token_tensor.shape)
print(token_tensor.dtype)

embedding = nn.Embedding(len(word_to_id), 3)
print(embedding)

vectors = embedding(token_tensor)
print(vectors)
print(vectors.shape)

print (vectors[0] == vectors[3])

print(embedding.weight)
print(embedding.weight.shape)
print(embedding.weight[0] == vectors[0])

input_ids = token_ids[:-1]
target_ids = token_ids[1:]

print(input_ids, target_ids)

input_tensor = torch.tensor(input_ids, dtype = torch.long)
target_tensor = torch.tensor(target_ids, dtype = torch.long)

print(input_tensor.shape, target_tensor.shape)

input_vectors = embedding(input_tensor)
print(input_vectors)
print(input_vectors.shape)

output_layer = nn.Linear(3, len(word_to_id))
scores = output_layer(input_vectors)
print(scores, scores.shape)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(scores, target_tensor)
print(loss.item())

embedding_params = list(embedding.parameters())
output_params = list(output_layer.parameters())
params = embedding_params + output_params
optimizer = torch.optim.Adam(params, lr = 0.05)
print(len(params))

total_params = 0
for param in params:
    total_params += param.numel()
print(total_params)

epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    input_vectors = embedding(input_tensor)
    scores = output_layer(input_vectors)
    loss = loss_fn(scores, target_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(
            "Эпоха:", epoch + 1,
            "Значение ошибки:", loss.item()
        )

with torch.no_grad():
    input_vectors = embedding(input_tensor)
    scores = output_layer(input_vectors)
    predict_ids = torch.argmax(scores, dim = 1)
print(predict_ids, target_tensor)

id_to_word = {}
for  word, id in word_to_id.items():
        id_to_word[id] = word
print(id_to_word)

prediction_words = []
for token_id in predict_ids:
    word = id_to_word[token_id.item()]
    prediction_words.append(word)
print(prediction_words)


start_word = 'кот'
start_id = word_to_id[start_word]
start_tensor = torch.tensor([start_id], dtype = torch.long)
with torch.no_grad():
    vectors = embedding(start_tensor)
    scores = output_layer(vectors)
    next_id = torch.argmax(scores, dim = 1)
next_word = id_to_word[next_id.item()]
print(start_word, '->' , next_word)

generated_words = ['кот']
current_word = 'кот'
for step in range(5):
    current_id = word_to_id[current_word]
    current_tensor = torch.tensor([current_id], dtype = torch.long)
    with torch.no_grad():
        vectors = embedding(current_tensor)
        scores = output_layer(vectors)
        next_id = torch.argmax(scores, dim = 1)
        next_word = id_to_word[next_id.item()]
        generated_words.append(next_word)
        current_word = next_word
print(' '.join(generated_words))

class NextWordModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(len(word_to_id), 3)
        self.output = nn.Linear(3,len(word_to_id))
    def forward(self, x):
        vectors = self.embedding(x)
        scores = self.output(vectors)
        return scores
model = NextWordModel()
scores = model(input_tensor)
print(scores.shape)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    scores = model(input_tensor)
    loss = loss_fn(scores, target_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print('Эпоха ', epoch + 1,
              'номер ошибки ', loss.item())
model.eval()
with torch.no_grad():
    scores = model(input_tensor)
    id = torch.argmax(scores, dim = 1)
print(id, target_tensor)

model_state = model.state_dict()
torch.save(model_state, "next_word_model.pth")
torch.save(word_to_id, 'vocab.pth')
print('модель сохранена')
