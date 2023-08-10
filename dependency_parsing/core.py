import os
import json
import torch

from neuronlp2.io import  conllx_data, conllu_data
from neuronlp2.models import DeepBiAffine

ROOT_CHAR = "_ROOT_CHAR"
ROOT_POS = "_ROOT_POS"
ROOT = "[CLS]"

class DependencyParser: 
    def __init__(self):
        pwd = os.getcwd()
        alphabet_path = "/workspace/Tugas Akhir/indolem/dependency_parsing/experiments/scripts/models/indobertGSD/alphabets"
        self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet = conllx_data.create_alphabets(alphabet_path, None)
        
        model_dir = "/workspace/Tugas Akhir/indolem/dependency_parsing/experiments/scripts/models/indobertGSD/model.pt"
        arg_path = "/workspace/Tugas Akhir/indolem/dependency_parsing/experiments/scripts/models/indobertGSD/config.json"
        
        hyps = json.load(open(arg_path, 'r'))
        
        self.model_path = "/workspace/Tugas Akhir/indolem/dependency_parsing/experiments/scripts/models/indobertGSD/model.pt"
        
        word_dim = hyps['word_dim']
        char_dim = hyps['char_dim']
        use_pos = hyps['pos']
        pos_dim = hyps['pos_dim']
        mode = hyps['rnn_mode']
        hidden_size = hyps['hidden_size']
        arc_space = hyps['arc_space']
        type_space = hyps['type_space']
        p_in = hyps['p_in']
        p_out = hyps['p_out']
        p_rnn = hyps['p_rnn']
        activation = hyps['activation']
        prior_order = None
        num_words = self.word_alphabet.size()
        num_chars = self.char_alphabet.size()
        num_pos = self.pos_alphabet.size()
        num_types = self.type_alphabet.size()
        num_layers = hyps['num_layers']
        self.model = DeepBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                               mode, hidden_size, num_layers, num_types, arc_space, type_space,
                               p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
        
        self.model.load_state_dict(torch.load(self.model_path))
    
    def parse_rows(self, rows):
        modified_rows = rows.copy()
        hasil, heads_pred, types_pred = self.predict(rows)
        
        # i = 1
        # for row in modified_rows:
        #     if (row[0].isnumeric()):
        #         row[6] = str(heads_pred[i])
        #         row[7] = self.type_alphabet.get_instance(types_pred[i])
        #         i += 1
        
        return hasil
        # return modified_rows

    def predict(self, rows):
        # self.model.cpu()
        device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.model.eval()
        
        words, chars, postags, masks = self.convert_to_tensor(rows)
        
        with torch.no_grad():
            temp_heads_pred, temp_types_pred = self.model.decode(words.to(device), chars.to(device), postags.to(device), mask=masks.to(device), leading_symbolic=conllu_data.NUM_SYMBOLIC_TAGS)

        heads_pred = temp_heads_pred[0, :]
        types_pred = temp_types_pred[0, :]
        w = self.word_alphabet.get_instance(words[0, 1])
        t = self.type_alphabet.get_instance(types_pred[1])
        h = heads_pred[1]
        
        hasil = []
        count = 0
        for i, j, k in zip(words[0, 1:],types_pred[1:],heads_pred[1:]):
            w = self.word_alphabet.get_instance(i)
            t = self.type_alphabet.get_instance(j)
            if w == "<_UNK>":
                w = rows[count][0]
            hasil.append((count, w, t, k-1))
            count += 1
            
        
        return hasil, heads_pred, types_pred
    
    def convert_to_tensor(self, rows):
        data, _ = conllu_data.read_data(rows, self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet, symbolic_root=True)
        words = data['WORD']
        chars = data['CHAR']
        postags = data['POS']
        masks = data['MASK']
        
        return words, chars, postags, masks

