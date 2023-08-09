def open_file_pos_tag(path):
    with open(path, "r", encoding="unicode_escape") as f:
        data = []
        sentence = []
        for line in f:
            if line.strip() == "":
                if len(sentence) > 0:
                    data.append(sentence)
                    sentence = []
            else:
                word, pos = line.strip().split("\t")
                sentence.append((word, pos))
        if len(sentence) > 0:
            data.append(sentence)
    return data
