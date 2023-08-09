def word2features(sent, i):
    word = sent[i][0]
    features = {
        "word": word,
        "bias": 1.0,
    }
    if i > 0:
        prev_word = sent[i - 1][0]
        features.update(
            {
                "prev_word": prev_word,
                "prev_word.lower()": prev_word.lower(),
                "prev_word.isupper()": prev_word.isupper(),
                "prev_word.istitle()": prev_word.istitle(),
                "prev_word.isdigit()": prev_word.isdigit(),
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        next_word = sent[i + 1][0]
        features.update(
            {
                "next_word": next_word,
                "next_word.lower()": next_word.lower(),
                "next_word.isupper()": next_word.isupper(),
                "next_word.istitle()": next_word.istitle(),
                "next_word.isdigit()": next_word.isdigit(),
            }
        )
    else:
        features["EOS"] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]
