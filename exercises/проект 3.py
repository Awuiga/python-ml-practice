sentences = [
    "python learns from data",
    "python learns very fast",
    "python code can be simple",
    "machine learning uses data",
    "machine learning can be useful",
]

def prepare_sentences(sentences):
    prepared = []
    for sentence in sentences:
        prepared.append(sentence.split())
    return prepared
print(prepare_sentences(sentences))

def train_autocomplete(prepared_sentences):
    counts = {}

    for sentence in prepared_sentences:
        for word in range(len(sentence) - 1):
            current_word = sentence[word]
            next_word = sentence[word + 1]

            if current_word not in counts:
                counts[current_word] = {}
            if next_word not in counts[current_word]:
                counts[current_word][next_word] = 0
            counts[current_word][next_word] += 1
    return counts

def suggest_next_word(current_word, counts):
    if current_word not in counts:
        return None

    result = counts[current_word]

    best_word = None
    best_count = -1

    for word, count in result.items():
        if count > best_count:
            best_count = count
            best_word = word
    return best_word
prepared = prepare_sentences(sentences)
counts = train_autocomplete(prepared)

print(suggest_next_word("python", counts))

def autocomplete_text(start_text, amount, counts):
    result = start_text.split()

    for _ in range(amount):
        current_word = result[-1]
        next_word = suggest_next_word(current_word, counts)

        if next_word is None:
            break

        result.append(next_word)

    return ' '.join(result)
