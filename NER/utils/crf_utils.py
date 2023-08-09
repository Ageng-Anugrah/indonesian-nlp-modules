def word2features_ner(df, i):
    word = df[i][0]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }
    if i > 0:
        prev_word = df[i-1][0]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()": prev_word.isupper(),
        })
    else:
        features["BOS"] = True
    if i < len(df)-1:
        next_word = df[i+1][0]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isupper()": next_word.isupper(),
        })
    else:
        features["EOS"] = True
    return features

def sent2features_ner(sent):
    return [word2features_ner(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]