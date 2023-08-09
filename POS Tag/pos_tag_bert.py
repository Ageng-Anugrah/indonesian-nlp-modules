import os

# Change to available GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import nltk
import numpy as np
import sklearn_crfsuite
import torch
from sklearn_crfsuite import metrics
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast,
)
from utils.bert_utils import dataset
from utils.crf_utils import sent2features, sent2labels
from utils.file_utils import open_file_pos_tag
from utils.pos_utils import remove_bio_tags

# File format must be conll or conllu
train_data_path = ""
val_data_path = ""
test_data_path = ""

# Read POS tagged data
train_data = open_file_pos_tag(train_data_path)
val_data = open_file_pos_tag(val_data_path)
test_data = open_file_pos_tag(test_data_path)

base_model = "model/indobert-large-p2-finetuned-pos"


unique_labels = list(
    set([lab for label in train_labels + val_labels + test_labels for lab in label])
)
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

device = "cuda" if cuda.is_available() else "cpu"

model.to(device)
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
