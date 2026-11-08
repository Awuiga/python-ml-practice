import random
names = [
    "anna",
    "alex",
    "maria",
    "mark",
    "max",
    "diana",
    "daniel",
    "lina",
    "leo",
    "nina",
]

def get_chars(names):
    chars = []

    for name in names:
        for char in name:
            chars.append(char)
    return sorted(set(chars))
print(get_chars(names))

def make_char_to_id(chars):
    char_to_id = {}

    for char_id, char in enumerate(chars):
        char_to_id[char] = char_id
    return char_to_id

def make_id_to_char(chars):
    id_to_char = {}
    for char_id, char in enumerate(chars):
        id_to_char[char_id] = char
    return id_to_char

def encode_name(name, char_to_id):
    result = []

    for char in name:
        result.append(char_to_id[char])
    return result

def decode_name(ids, id_to_char):
    result = []

    for char_id in ids:
        result.append(id_to_char[char_id])
    return ''.join(result)

def prepare_name(name):
    return '^' + name +'$'
print(prepare_name("anna"))

def prepare_names(names):
    prepared = []

    for name in names:
        prepared.append(prepare_name(name))
    return prepared
print(prepare_names(names))

def train_names(prepared_names):
    counts = {}

    for name in prepared_names:
        for i in range(len(name) - 1):
            current = name[i]
            next_char = name[i + 1]

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

def generate_name(counts):
    result = '^'
    current = '^'

    for _ in range(20):
        next_char = choose_next_char(current, counts)
        if next_char == '$':
            break

        result += next_char
        current = next_char
    return result[1:]

def generate_names(counts, amount):
    names = []

    for _ in range(amount):
        name = generate_name(counts)
        names.append(name)
    return names

prepared_names = prepare_names(names)
counts = train_names(prepared_names)

print(generate_names(counts, 10))
