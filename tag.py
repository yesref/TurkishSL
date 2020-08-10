
from input.embeddings import read_embeddings_file
from input.read_dataset import read_data_files
from util.evaluation import compute_f1, draw_confusion_matrix
from util.logger import get_logger
from util.util import get_max_word_morpheme_dim
from util.preprocess import get_data_ids, get_batch_lengths, iterate_batches
from model.models import TaggingModel

import numpy as np
import pandas as pd
import seaborn as sn
import configargparse
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from keras.utils import to_categorical

logger = get_logger('Tagging')

parser = configargparse.get_argument_parser('Parameters')
parser.add_argument('-f', '--config_file', required=True, is_config_file=True, help='config file path')
parser.add_argument('-d', '--debug_mode', help='debug mode', action='store_true', default=False)
parser.add_argument('-p', '--problem_type', help='sequence problem tpe', choices=['NER', 'POS'])
parser.add_argument('--train_file', help='path for training data', required=True)
parser.add_argument('--dev_file', help='path for validation data', required=True)
parser.add_argument('--test_file', help='path for test data', required=True)
parser.add_argument('--embedding_file', help='path for pre embedding vectors file', required=True)
parser.add_argument('--embedding_type', choices=['huawei', 'fastText', 'glove'], default='huawei')
parser.add_argument('--epochs', help='number of epoch', default=50, type=int)
parser.add_argument('--executions', help='execution count', default=1, type=int)
parser.add_argument('--cnn_dropouts', nargs=2, type=float, help='dropout values for cnn layer and char embeddings')
parser.add_argument('--rnn_dropout', type=float, help='dropout value for rnn layer')
parser.add_argument('--recurrent_dropout', type=float, help='dropout value for recurrent hidden vectors')
parser.add_argument('--rnn_type', choices=['LSTM', 'GRU'], default='LSTM')
parser.add_argument('--use_morpheme', help='whether morpheme features are used', action='store_true', default=False)
parser.add_argument('--morph_dropout', type=float, help='dropout value for morpheme rnn layer features')
parser.add_argument('--morph_state_size', type=int, help='number of hidden state in morpheme level rnn', default=50)
parser.add_argument('--add_morph_att', help='whether morpheme attention is used', action='store_true', default=False)
parser.add_argument('--use_morph_self_att', help='use self attention or not', action='store_true', default=False)
parser.add_argument('--self_att_width', type=int, help='width of self attention layer for neighbour words', default=10)
parser.add_argument('--self_att_dropout', type=float, help='dropout value for self attention layer of morphemes')
parser.add_argument('--self_att_concat_pos', choices=['BeforeRNN', 'BeforeCRF'], default='BeforeRNN')
parser.add_argument('--use_cnn', help='whether character features are used or not', action='store_true', default=False)
parser.add_argument('--use_crf', help='whether crf is used or not', action='store_true', default=False)

conf = parser.parse_known_args()[0]
logger.info("The program has started with the configuration in " + conf.config_file + " file")
print(parser.format_values())

# read data from data files and with using data generate particular alphabets
alphabets, labels, sentences, morphemes = read_data_files(conf.train_file, conf.dev_file, conf.test_file)
word_alphabet, char_alphabet, label_alphabet, morph_alphabet = alphabets
train_sentences, dev_sentences, test_sentences = sentences
train_labels, dev_labels, test_labels = labels
train_morphemes, dev_morphemes, test_morphemes = morphemes

max_word_length = len(max(word_alphabet.get_content()['instances'], key=len))
max_morph_dim = get_max_word_morpheme_dim(train_morphemes, dev_morphemes, test_morphemes)
logger.info("Alphabets were created and datasets were read")
logger.info("The length of longest word in dataset is: %d" % max_word_length)
logger.info("The maximum morpheme count take part in one word in dataset is: %d" % max_morph_dim)

# with using generated alphabets convert data lists to new structure to hold numbers instead of characters and strings
train_dataset = get_data_ids(alphabets, train_sentences, train_labels, train_morphemes, max_morph_dim)
dev_dataset = get_data_ids(alphabets, dev_sentences, dev_labels, dev_morphemes, max_morph_dim)
test_dataset = get_data_ids(alphabets, test_sentences, test_labels, test_morphemes, max_morph_dim)
logger.info("Words, Characters and Labels in datasets were converted to their number equivalents with using alphabets")

if conf.debug_mode:
    epochs = 1
    train_dataset = train_dataset[:1000]
    dev_dataset = dev_dataset[:250]
    test_dataset = test_dataset[:250]
    embeddings, dimension = {}, 300
else:
    epochs = conf.epochs
    embeddings, dimension = read_embeddings_file(word_alphabet, conf.embedding_file)
    logger.info("Word embeddings were read from file")

# separate dataset to batches according to number of words in sentences. sentence with same # of words be in same batch
train_dataset, train_batch_lengths = get_batch_lengths(train_dataset)
dev_dataset, dev_batch_lengths = get_batch_lengths(dev_dataset)
test_dataset, test_batch_lengths = get_batch_lengths(test_dataset)

# create a subset of embeddings with respect to current dataset
embeddings = [embeddings[word] if word in embeddings
              else embeddings[word.lower()] if word.lower() in embeddings
              else np.random.uniform(-0.25, 0.25, dimension)
              for word in word_alphabet.get_content()['instances']]
embeddings = np.array(embeddings).astype(np.float)
embeddings = np.vstack((np.random.uniform(-0.25, 0.25, dimension).reshape(1, dimension), embeddings))
logger.info("A sub embedding table was extracted from original embeddings for studying datasets")
logger.info("Number of words in new embeddings table: %d" % embeddings.shape[0])

tagging_model = TaggingModel(embeddings, max_word_length, max_morph_dim, morph_alphabet.size(), char_alphabet.size(),
                             label_alphabet, conf.add_morph_att, conf.use_morpheme, conf.use_crf, conf.use_cnn,
                             conf.rnn_type, conf.cnn_dropouts, conf.rnn_dropout, conf.recurrent_dropout,
                             conf.morph_dropout, conf.morph_state_size, conf.use_morph_self_att, conf.self_att_width,
                             conf.self_att_dropout, conf.self_att_concat_pos)
tagging_model.compile('rmsprop')
logger.info("The model was created")
tagging_model.get_model().summary()

all_results = []
for j in range(conf.executions):

    losses_train, losses_dev = [], []

    # train model with using training batches
    for epoch in range(1, epochs+1):
        loss_train, loss_dev = [], []

        for i, batch in enumerate(tqdm(iterate_batches(train_dataset, train_batch_lengths),
                                       desc=str.format("Train Epoch %d/%d: " % (epoch, epochs)))):
            words, chars, morphs, labels = batch
            labels = [to_categorical(label, num_classes=label_alphabet.size()) for label in labels]
            morphs = to_categorical(morphs, num_classes=morph_alphabet.size())
            loss_train.append(tagging_model.train(conf.use_cnn, conf.use_morpheme, words, chars, morphs, labels))

        for i, batch in enumerate(tqdm(iterate_batches(dev_dataset, dev_batch_lengths),
                                       desc=str.format("Validation Epoch %d/%d: " % (epoch, epochs)))):
            words, chars, morphs, labels = batch
            labels = [to_categorical(label, num_classes=label_alphabet.size()) for label in labels]
            morphs = to_categorical(morphs, num_classes=morph_alphabet.size())
            loss_dev.append(tagging_model.validate(conf.use_cnn, conf.use_morpheme, words, chars, morphs, labels))

        losses_train.append(loss_train)
        losses_dev.append(loss_dev)

    # plt.plot([sum(losses)/len(losses) for losses in losses_train])
    # plt.plot([sum(losses)/len(losses) for losses in losses_dev])
    # plt.title('Model Hata Eğrisi')
    # plt.ylabel('Hata (Loss)')
    # plt.xlabel('Devir (Epoch)')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    tagging_model.get_model().save('sl_model.h5')

    correct_labels = []
    pred_labels = []

    # evaluate the model
    for i in trange(len(test_dataset)):
        words, chars, morphs, labels = test_dataset[i]
        morphs = to_categorical(np.asarray([morphs]), num_classes=morph_alphabet.size())
        pred = tagging_model.predict(conf.use_cnn, conf.use_morpheme, np.asarray([words]), np.asarray([chars]), morphs)
        correct_labels.append(labels)
        pred_labels.append(pred.tolist())

    all_results.append(compute_f1(conf.problem_type, pred_labels, correct_labels, label_alphabet))
    logger.info('Execution ' + str(j+1) + ' has finished')

    stc_orj, corr_hata_analysis, pred_hata_analysis = [], [], []
    for i in range(len(correct_labels)):
        if correct_labels[i] != pred_labels[i]:
            corr_hata_analysis.append(correct_labels[i])
            pred_hata_analysis.append(pred_labels[i])
            stc_orj.append(test_dataset[i][0])

    if conf.problem_type == 'NER':
        cm = draw_confusion_matrix(label_alphabet, pred_labels, correct_labels)

        cm_lbl = ['O', 'PERSON', 'ORGANIZATION', 'LOCATION']
    else:
        cm_dim = label_alphabet.size()-2
        cm_lbl = label_alphabet.get_content()['instances'][1:-1]

        cm = np.zeros((cm_dim, cm_dim))
        for i in range(len(pred_labels)):
            for j in range(len(pred_labels[i])):
                cm[correct_labels[i][j] - 1][pred_labels[i][j] - 1] += 1

    cm_sum = np.sum(cm, axis=1)
    cm_avg = (cm.T * 100 / cm_sum).T

    cm_df = pd.DataFrame(cm, index=cm_lbl, columns=cm_lbl)
    cm_df_avg = pd.DataFrame(cm_avg, index=cm_lbl, columns=cm_lbl)

    plt.figure(figsize=(10, 7))
    sn.heatmap(cm_df, annot=True, cmap="YlGnBu", fmt='.0f', vmin=0, vmax=500, annot_kws={"size": 16})
    plt.ylabel('Doğru Etiketler')
    plt.xlabel('Tahmin Edilen Etiketler')
    plt.show()

    plt.figure(figsize=(10, 7))
    sn.heatmap(cm_df_avg, annot=True, cmap="YlGnBu", fmt='.2f', vmin=0, vmax=100, annot_kws={"size": 16})
    plt.ylabel('Doğru Etiketler')
    plt.xlabel('Tahmin Edilen Etiketler')
    plt.show()

for results in all_results:
    if conf.problem_type == 'NER':
        pre_test, rec_test, f1_test = results
        print("Test-Data: Prec: %.4f, Rec: %.4f, F1: %.4f" % (pre_test, rec_test, f1_test))
    else:
        print("Test-Data: Accuracy: %.4f" % results)
