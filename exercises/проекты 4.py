reviews = [
    ("good movie", "positive"),
    ("great film", "positive"),
    ("good acting", "positive"),
    ("bad movie", "negative"),
    ("terrible film", "negative"),
    ("bad acting", "negative"),
]

def prepare_reviews(reviews):
    prepared = []

    for text, label in reviews:
        tokens = text.split()

        prepared.append((tokens, label))
    return prepared
print(prepare_reviews(reviews))

def train_word_counts(prepared_reviews):
    counts = {
        'positive': {},
        'negative': {}
    }

    for tokens, label in prepared_reviews:
        for word in tokens:
            if word not in counts[label]:
                counts[label][word] = 0
            counts[label][word] += 1
    return counts

def score_review(tokens, counts):
    positive_score = 0
    negative_score = 0

    for word in tokens:
        if word in counts['positive']:
            positive_score += counts["positive"][word]
        if word in counts['negative']:
            negative_score += counts["negative"][word]
    return positive_score, negative_score


def classify_review(tokens, counts):
    positive_score, negative_score = score_review(tokens, counts)
    if positive_score > negative_score:
        return 'positive'
    elif negative_score > positive_score:
        return 'negative'
    else:
        return 'unknown'

prepared_reviews = prepare_reviews(reviews)
counts = train_word_counts(prepared_reviews)

print(classify_review(["good", "movie"], counts))
print(classify_review(["bad", "acting"], counts))
print(classify_review(["movie"], counts))
