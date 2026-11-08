import torch
import torch.nn as nn


def make_context_pairs(token_ids, context_size):
    inputs = []
    targets = []
    for i in range(len(token_ids) - context_size):
        context = token_ids[i:i +context_size]
        next_word = token_ids[i + context_size]
        inputs.append(context)
        targets.append(next_word)
    return inputs, targets

token_ids = [0, 1, 2, 0, 3]
context_size = 2

inputs, targets = make_context_pairs(token_ids, context_size)

print(inputs)
print(targets)

input_tensor = torch.tensor(inputs, dtype = torch.long)
target_tensor = torch.tensor(targets, dtype = torch.long)
embedding = nn.Embedding(5, 3)
vectors = embedding(input_tensor)
combined = vectors.flatten(start_dim = 1)
print(input_tensor, vectors, combined, target_tensor)

output_layer = nn.Linear(6, len(set(token_ids)))
scores = output_layer(combined)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(scores, target_tensor)
print(scores.shape, loss.item())

class ContextLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(context_size * embedding_dim, vocab_size)

    def forward(self, x):
        vectors = self.embedding(x)
        combined = vectors.flatten(start_dim = 1)
        scores = self.output(combined)
        return scores

model = ContextLanguageModel(4,2,3)
model(input_tensor)
print(model(input_tensor).shape)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)
epochs = 500

for epoch in range(epochs):
    optimizer.zero_grad()
    scores = model(input_tensor)
    loss = loss_fn(scores, target_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print("Эпоха", epoch + 1,
              "Ошибка", loss.item())

model.eval()
generated_ids = [0,1]

for step in range(3):
    current_ids = generated_ids[-2:]
    current_tensor = torch.tensor([current_ids], dtype = torch.long)
    with torch.no_grad():
        scores = model(current_tensor)
        next_id = torch.argmax(scores, dim = 1)
        generated_ids.append(next_id.item())
print(generated_ids)
