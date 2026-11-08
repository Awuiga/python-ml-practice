import random
text = "banana"
def get_chars(text):
    result = sorted(set(text))
    return result
print(get_chars(text))

def make_char_to_id(chars):
    counts = {}
    for i, char in enumerate(chars):
        counts[char] = i
    return counts

chars = get_chars(text)
print(make_char_to_id(chars))

def make_id_to_char(chars):
    id_counts = {}
    for i, char in enumerate(chars):
        id_counts[i] = char
    return id_counts
print(make_id_to_char(chars))

char_to_id = make_char_to_id(chars)

def encode(text, char_to_id):
    result = []
    for char in text:
        result.append(char_to_id[char])
    return result

id_to_char = make_id_to_char(chars)
def decode(ids, id_to_char):
    result = []
    for i in ids:
        result.append(id_to_char[i])
    return ''.join(result)
print(encode(text, char_to_id))
encoded = encode(text, char_to_id)

print(encoded)
print(decode(encoded, id_to_char))


def train(text, context_size):
    counts = {}
    for i in range(len(text) - context_size):
        current = text[i:i + context_size]
        next_char = text[i + context_size]

        if current not in counts:
            counts[current] = {}
        if next_char not in counts[current]:
            counts[current][next_char] = 0
        counts[current][next_char] += 1
    return counts


def choose_next_char(current, counts):
    options = list(counts[current].keys())
    weights = list(counts[current].values())


    next_char = random.choices(
        options,
        weights = weights,
        k = 1
    )[0]

    return next_char
import random

def generate(start_text, length, counts, context_size):
    result = start_text
    current = start_text[-context_size:]

    for _ in range(length - len(start_text)):
        if current not in counts:
            break

        next_char = choose_next_char(current,counts)
        result += next_char
        current = result[-context_size:]
    return result



def get_probabilities(current, counts):
    transitions = counts[current]

    total = sum(transitions.values())

    probalities = {}

    for next_char, count in transitions.items():
        probality = count / total
        probalities[next_char] = probality
    return probalities

counts = train("ababaca", 1)

print(get_probabilities("a", counts))

def predict_next_char(current, counts):
    probabilities = get_probabilities(current, counts)
    best_char = None
    best_probality = -1
    for char, probability in probabilities.items():
        if probability > best_probality:
            best_probality = probability
            best_char = char
    return best_char

def generate_best(start_text, length, counts, context_size):
    result = start_text
    current = start_text[-context_size:]

    for _ in range(length - len(start_text)):
        next_char = predict_next_char(current, counts)
        result += next_char
        current = result[-context_size:]
    return result

text = "ababaca"
counts = train(text, 1)
print(generate("a", 15, counts, 1))
counts = train(text, 2)
print(generate('ab', 15, counts, 2 ))
counts = train(text, 3)
print(generate('aba', 15, counts, 3 ))
