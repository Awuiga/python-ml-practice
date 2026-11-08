import random
text = "python learns from text python learns fast"

def tokenize(text):
    return text.split()

def get_vocab(tokens):
    return sorted(set(tokens))
tokens = tokenize(text)
vocab = get_vocab(tokens)

print(tokens)
print(vocab)

def make_word_to_id(vocab):
    id_counts = {}
    for word_id, word in enumerate(vocab):
        id_counts[word] = word_id
    return id_counts

def make_id_to_word(vocab):
    word_counts = {}

    for word_id, word in enumerate(vocab):
        word_counts[word_id] = word

    return word_counts

def encode_words(tokens, word_to_id):
    result = []
    for word in tokens:
        result.append(word_to_id[word])
    return result

def decode_words(ids, id_to_word):
    result = []
    for word_id in ids:
        result.append(id_to_word[word_id])
    return result

def detokenize(tokens):
    return ' '.join(tokens)

word_to_id = make_word_to_id(vocab)
id_to_word = make_id_to_word(vocab)

encoded = encode_words(tokens, word_to_id)

print(encoded)
print(decode_words(encoded, id_to_word))

def train_words(tokens, context_size):
    counts = {}

    for i in range(len(tokens) - context_size):
        current_word = tuple(tokens[i:i + context_size])
        next_word = tokens[i + context_size]

        if current_word  not in counts:
            counts[current_word ] = {}

        if next_word not in counts[current_word ]:
            counts[current_word ][next_word] = 0
        counts[current_word][next_word] += 1
    return counts

counts = train_words(tokens, 1)
print(train_words(tokens, 1))
print(train_words(tokens, 2))

def choose_next_word(current_words, counts):
    options = list(counts[current_words].keys())
    weights = list(counts[current_words].values())

    next_word = random.choices(
        options,
        weights = weights,
        k = 1
    )[0]

    return next_word

def generate_words(start_words, length, counts, context_size):
    current = tuple(start_words[-context_size:])
    result = start_words.copy()

    for _ in range(length - len(start_words)):
        if current not in counts:
            break

        next_word = choose_next_word(current, counts)
        result.append(next_word)
        current = tuple(result[-context_size:])
    return result

counts = train_words(tokens, 2)

generated = generate_words(
    ["python", "learns"],
    7,
    counts,
    2
)

print(generated)
print(detokenize(generated))

def get_word_probabilities(current_words, counts):
    transitions = counts[current_words]

    total = sum(transitions.values())

    probabilities = {}

    for word, count in transitions.items():
        probability = count / total
        probabilities[word] = probability
    return probabilities

counts = train_words(tokens, 2)

print(
    get_word_probabilities(
        ("python", "learns"),
        counts
    )
)

def predict_next_word(current_words, counts):
    probabilities = get_word_probabilities(current_words, counts)

    best_word = None
    best_probability = -1

    for word, probability in probabilities.items():
        if probability > best_probability:
            best_probability = probability
            best_word = word
    return best_word

def generate_best_words(start_words, length, counts, context_size):
    result = start_words.copy()
    current = tuple(result[-context_size:])

    for _ in range(length - len(start_words)):
        if current not in counts:
            break

        next_word = predict_next_word(current, counts)
        result.append(next_word)
        current = tuple(result[-context_size:])
    return result
def make_zero_matrix(size):
    matrix = []

    for _ in range(size):
        row = []
        for _ in range(size):
            row.append(0)

        matrix.append(row)
    return matrix

def build_transition_matrix(encoded, vocab_size):
    matrix = make_zero_matrix(vocab_size)

    for i in range(len(encoded) - 1):
        current_id = encoded[i]
        next_id = encoded[i + 1]

        matrix[current_id][next_id] += 1

    return matrix

matrix = build_transition_matrix(encoded, len(vocab))

for row in matrix:
    print(row)

def row_to_probabilities(row):
    total = sum(row)

    probabilities = []

    for count in row:
        probability = count / total
        probabilities.append(probability)
    return probabilities

def matrix_to_probabilities(matrix):
    probability_matrix = []

    for row in matrix:
        if sum(row) == 0:
            probabilities = [0] * len(row)
        else:
            probabilities = row_to_probabilities(row)

        probability_matrix.append(probabilities)
    return probability_matrix

def choose_next_id(current_id, probability_matrix):
    probabilities = probability_matrix[current_id]

    if sum(probabilities) == 0:
        return None

    options = list(range(len(probabilities)))

    next_id = random.choices(
        options,
        weights=probabilities,
        k=1
    )[0]

    return next_id

def generate_ids(start_id, length, probability_matrix):
    result = [start_id]
    current_id = start_id

    for _ in range(length - 1):
        next_id = choose_next_id(current_id, probability_matrix)

        if next_id is None:
            break

        result.append(next_id)
        current_id = next_id
    return result

def generate_text(start_word, length, word_to_id, id_to_word, probability_matrix):
    start_id = word_to_id[start_word]

    generated_ids = generate_ids(start_id,length,probability_matrix)
    generated_words = decode_words(generated_ids, id_to_word)

    text = detokenize(generated_words)
    return text

tokens = tokenize(text)
vocab = get_vocab(tokens)

word_to_id = make_word_to_id(vocab)
id_to_word = make_id_to_word(vocab)

encoded = encode_words(tokens, word_to_id)


matrix = build_transition_matrix(
    encoded,
    len(vocab)
)

probability_matrix = matrix_to_probabilities(
    matrix
)

generated_text = generate_text(
    "python",
    10,
    word_to_id,
    id_to_word,
    probability_matrix
)

print(generated_text)
