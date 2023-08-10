"""
Implementation of Graph-based dependency parsing.
"""

import os
import sys
import gc
import json

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import argparse
import math
import numpy as np
import torch
from torch.optim.adamw import AdamW
from torch.optim import SGD
from torch.nn.utils import clip_grad_norm_
from neuronlp2.nn.utils import total_grad_norm
from neuronlp2.io import get_logger, conllu_data, iterate_data
from neuronlp2.models import DeepBiAffine, NeuroMST, StackPtrNet
from neuronlp2.optim import ExponentialScheduler
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser
from neuronlp2.nn.utils import freeze_embedding

def eval(alg, data, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, device, beam=1, batch_size=256):
    network.eval()
    accum_ucorr = 0.0
    accum_lcorr = 0.0
    accum_total = 0
    accum_ucomlpete = 0.0
    accum_lcomplete = 0.0
    accum_ucorr_nopunc = 0.0
    accum_lcorr_nopunc = 0.0
    accum_total_nopunc = 0
    accum_ucomlpete_nopunc = 0.0
    accum_lcomplete_nopunc = 0.0
    accum_root_corr = 0.0
    accum_total_root = 0.0
    accum_total_inst = 0.0
    for data in iterate_data(data, batch_size):
        print("LMAOO", data['WORD'])
        words = data['WORD'].to(device)
        chars = data['CHAR'].to(device)
        postags = data['POS'].to(device)
        heads = data['HEAD'].numpy()
        types = data['TYPE'].numpy()
        lengths = data['LENGTH'].numpy()
        print(words)
        if alg == 'graph':
            masks = data['MASK'].to(device)
            heads_pred, types_pred = network.decode(words, chars, postags, mask=masks, leading_symbolic=conllu_data.NUM_SYMBOLIC_TAGS)
        else:
            masks = data['MASK_ENC'].to(device)
            heads_pred, types_pred = network.decode(words, chars, postags, mask=masks, beam=beam, leading_symbolic=conllu_data.NUM_SYMBOLIC_TAGS)

        words = words.cpu().numpy()
        postags = postags.cpu().numpy()
        print("Ini dong", types_pred)
        print(pred_writer.write(words, postags, heads_pred, types_pred, lengths, symbolic_root=True))
        gold_writer.write(words, postags, heads, types, lengths, symbolic_root=True)

        stats, stats_nopunc, stats_root, num_inst = parser.eval(words, postags, heads_pred, types_pred, heads, types,
                                                                word_alphabet, pos_alphabet, lengths, punct_set=punct_set, symbolic_root=True)
        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        corr_root, total_root = stats_root

        accum_ucorr += ucorr
        accum_lcorr += lcorr
        accum_total += total
        accum_ucomlpete += ucm
        accum_lcomplete += lcm

        accum_ucorr_nopunc += ucorr_nopunc
        accum_lcorr_nopunc += lcorr_nopunc
        accum_total_nopunc += total_nopunc
        accum_ucomlpete_nopunc += ucm_nopunc
        accum_lcomplete_nopunc += lcm_nopunc

        accum_root_corr += corr_root
        accum_total_root += total_root

        accum_total_inst += num_inst

    print('W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr, accum_lcorr, accum_total, accum_ucorr * 100 / accum_total, accum_lcorr * 100 / accum_total,
        accum_ucomlpete * 100 / accum_total_inst, accum_lcomplete * 100 / accum_total_inst))
    print('Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr_nopunc, accum_lcorr_nopunc, accum_total_nopunc, accum_ucorr_nopunc * 100 / accum_total_nopunc,
        accum_lcorr_nopunc * 100 / accum_total_nopunc,
        accum_ucomlpete_nopunc * 100 / accum_total_inst, accum_lcomplete_nopunc * 100 / accum_total_inst))
    print('Root: corr: %d, total: %d, acc: %.2f%%' %(accum_root_corr, accum_total_root, accum_root_corr * 100 / accum_total_root))
    return (accum_ucorr, accum_lcorr, accum_ucomlpete, accum_lcomplete, accum_total), \
           (accum_ucorr_nopunc, accum_lcorr_nopunc, accum_ucomlpete_nopunc, accum_lcomplete_nopunc, accum_total_nopunc), \
           (accum_root_corr, accum_total_root, accum_total_inst)

def parse(args):
    logger = get_logger("Parsing")
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 5) if args.cuda else torch.device('cpu')
    test_path = args.test

    model_path = args.model_path
    model_name = os.path.join(model_path, 'model.pt')
    punctuation = args.punctuation
    print(args)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets')
    assert os.path.exists(alphabet_path)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllu_data.create_alphabets(alphabet_path, None)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    result_path = os.path.join(model_path, 'tmp')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    logger.info("loading network...")
    hyps = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    model_type = hyps['model']
    assert model_type in ['DeepBiAffine', 'NeuroMST', 'StackPtr']
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
    
    print(model_type, "Nyamm")

    alg = 'transition' if model_type == 'StackPtr' else 'graph'
    if model_type == 'DeepBiAffine':
        num_layers = hyps['num_layers']
        network = DeepBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                               mode, hidden_size, num_layers, num_types, arc_space, type_space,
                               p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    elif model_type == 'NeuroMST':
        num_layers = hyps['num_layers']
        network = NeuroMST(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                           mode, hidden_size, num_layers, num_types, arc_space, type_space,
                           p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    elif model_type == 'StackPtr':
        encoder_layers = hyps['encoder_layers']
        decoder_layers = hyps['decoder_layers']
        num_layers = (encoder_layers, decoder_layers)
        prior_order = hyps['prior_order']
        grandPar = hyps['grandPar']
        sibling = hyps['sibling']
        network = StackPtrNet(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                              mode, hidden_size, encoder_layers, decoder_layers, num_types, arc_space, type_space,
                              prior_order=prior_order, activation=activation, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                              pos=use_pos, grandPar=grandPar, sibling=sibling)
    else:
        raise RuntimeError('Unknown model type: %s' % model_type)

    network = network.to(device)
    network.load_state_dict(torch.load(model_name, map_location=device))
    model = "{}-{}".format(model_type, mode)
    logger.info("Network: %s, num_layer=%s, hidden=%d, act=%s" % (model, num_layers, hidden_size, activation))

    logger.info("Reading Data")
    if alg == 'graph':
        data_test = conllu_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True)
    else:
        raise Exception('Connlu data reader for StackPtrNet is not implemented')
    beam = args.beam
    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    pred_filename = os.path.join(result_path, 'pred.txt')
    pred_writer.start(pred_filename)
    gold_filename = os.path.join(result_path, 'gold.txt')
    gold_writer.start(gold_filename)
    print(result_path, "Cok")

    with torch.no_grad():
        print('Parsing...')
        start_time = time.time()
        eval(alg, data_test, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, device, beam, batch_size=args.batch_size)
        print('Time: %.2fs' % (time.time() - start_time))

    pred_writer.close()
    gold_writer.close()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--mode', choices=['train', 'parse'], required=True, help='processing mode')
    args_parser.add_argument('--config', type=str, help='config file')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    args_parser.add_argument('--loss_type', choices=['sentence', 'token'], default='sentence', help='loss type (default: sentence)')
    args_parser.add_argument('--optim', choices=['sgd', 'adam'], help='type of optimizer')
    args_parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    args_parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam')
    args_parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam')
    args_parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--lr_decay', type=float, default=0.999995, help='Decay rate of learning rate')
    args_parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    args_parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
    args_parser.add_argument('--warmup_steps', type=int, default=0, metavar='N', help='number of steps to warm up (default: 0)')
    args_parser.add_argument('--reset', type=int, default=10, help='Number of epochs to reset optimizer (default 10)')
    args_parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    args_parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot', 'fasttext', 'bert'], help='Embedding for words')
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--word2index_path', help='path for word2index for loading BERT embedding')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters')
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train', help='path for training file.')
    args_parser.add_argument('--dev', help='path for dev file.')
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--normalize_digits', type=int, default=0, help='Either 1 or 0')

    args = args_parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        parse(args)
