
import numpy as np
from tqdm import trange
from keras.models import load_model
from keras.utils import to_categorical
from keras_contrib.layers import CRF

from util.evaluation import compute_f1
from input.read_dataset import read_data_files
from util.util import get_max_word_morpheme_dim
from util.preprocess import get_data_ids, get_batch_lengths
from model.attention import VanillaAttention, SelfAttention


problem_type = 'POS'
data_file = 'pos'
path = 'data/' + data_file + '_'
use_morpheme = True
use_cnn = False


loaded_model = load_model('sl_model.h5', custom_objects={'VanillaAttention': VanillaAttention, 'SelfAttention': SelfAttention})

alphabets, labels, sentences, morphemes = read_data_files(path + 'train.txt', path + 'valid.txt', path + 'test.txt')
word_alphabet, char_alphabet, label_alphabet, morph_alphabet = alphabets
test_sentences = sentences[2]
test_labels = labels[2]
train_morphemes, dev_morphemes, test_morphemes = morphemes

max_morph_dim = get_max_word_morpheme_dim(train_morphemes, dev_morphemes, test_morphemes)
test_dataset = get_data_ids(alphabets, test_sentences, test_labels, test_morphemes, max_morph_dim)
test_dataset, test_batch_lengths = get_batch_lengths(test_dataset)

correct_labels = []
pred_labels = []

for i in trange(len(test_dataset)):
    words, chars, morphs, labels = test_dataset[i]
    morphs = to_categorical(np.asarray([morphs]), num_classes=morph_alphabet.size())

    words, chars, morphemes = np.asarray([words]), np.asarray([chars]), morphs
    model_input = [words, morphemes] if use_morpheme else [words]
    model_input = model_input + [chars] if use_cnn else model_input
    pred = loaded_model.predict(model_input)
    pred = pred[0].argmax(axis=-1)

    correct_labels.append(labels)
    pred_labels.append(pred.tolist())

results = compute_f1(problem_type, pred_labels, correct_labels, label_alphabet)

if problem_type == 'NER':
    pre_test, rec_test, f1_test = results
    print("Test-Data: Prec: %.4f, Rec: %.4f, F1: %.4f" % (pre_test, rec_test, f1_test))
else:
    print("Test-Data: Accuracy: %.4f" % results)

