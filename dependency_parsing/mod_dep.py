class DependencyParser: 
    def __init__(self):
        pwd = os.getcwd()
        alphabet_path = "alphabets_path"
        self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet = conllx_data.create_alphabets(alphabet_path, None)
        
        model_dir = "model_path"
        arg_path = "arg_path"
        
        hyps = json.load(open(arg_path, 'r'))
        
        ...
        
        self.model = DeepBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                               mode, hidden_size, num_layers, num_types, arc_space, type_space,
                               p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
        self.model.load_state_dict(torch.load(self.model_path))
        
def get_dependency(rows):
    a = DependencyParser()
    return a.parse_rows(rows)
    
def parse_rows(self, rows):
    modified_rows = rows.copy()
    hasil, heads_pred, types_pred = self.predict(rows)

    return hasil

def predict(self, rows):
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    self.model.to(device)
    self.model.eval()
    
    words, chars, postags, masks = self.convert_to_tensor(rows)
    with torch.no_grad():
            temp_heads_pred, temp_types_pred = self.model.decode(words.to(device), chars.to(device), postags.to(device), mask=masks.to(device), leading_symbolic=conllu_data.NUM_SYMBOLIC_TAGS)
            
    ...
    
    for i, j, k in zip(words[0, 1:],types_pred[1:],heads_pred[1:]):
            w = self.word_alphabet.get_instance(i)
            t = self.type_alphabet.get_instance(j)
            if w == "<_UNK>":
                w = rows[count][0]
            hasil.append((count, w, t, k-1))
            count += 1
            
        
        return hasil, heads_pred, types_pred

# conllu_data.py
def read_data(rows, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
              max_size=None, normalize_digits=False, symbolic_root=False, symbolic_end=False):
    data = []
    max_length = 0
    max_char_length = 0
    counter = 0
    reader = CoNLLUReaderModified([rows], word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    
    ...
    
# reader.py
class CoNLLUReaderModified(object):
    def __init__(self, sentences, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__sentences = sentences
        self.__cur_idx = 0
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet
    
    ...
    

    
