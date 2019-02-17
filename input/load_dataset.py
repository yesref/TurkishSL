import sys
import jpype as jp
from random import shuffle
from input.morpheme import TurkishMorphemeAnalyzer


def read_enemax_file(filename):

    f = open(filename, encoding="utf-8")

    sentences = []
    sentence_labels = []

    for line in f:
        if len(line) == 0 or line[0] == "\n":
            continue

        tokens = line.strip().split(' ')

        words = []
        labels = []
        label = 'O'
        label_prefix = ''

        for token in tokens:
            if token == '' or token == '<b_enamex':
                continue
            if token.startswith('TYPE="'):
                type_index_end = token.index('"', 6)
                label = token[6:type_index_end]
                label_prefix = 'B-'
                if token.endswith('e_enamex>'):
                    word = token[token.index('">')+2:token.index('<e_')]
                    words.append(word)
                    labels.append(label_prefix + label)
                    label_prefix = ''
                    label = 'O'
                else:
                    word = token[token.index('">')+2:]
                    if word == '':
                        continue
                    words.append(word)
                    labels.append(label_prefix + label)
                    label_prefix = 'I-'
            else:
                if token.endswith('e_enamex>'):
                    word = token[:token.index('<e_')]
                    if word != '':
                        words.append(word)
                        labels.append(label_prefix + label)
                    label_prefix = ''
                    label = 'O'
                else:
                    words.append(token)
                    labels.append(label_prefix + label)
                    if label_prefix == 'B-':
                        label_prefix = 'I-'

        sentences.append(" ".join(str(x) for x in words))
        sentence_labels.append(" ".join(str(x) for x in labels))

    return sentences, sentence_labels


def read_conll_file(filename):

    file = open(filename, encoding="utf-8")

    sentences = []
    sentences_labels = []
    sentence = []
    labels = []

    for line in file:
        if len(line) == 0 or line[0] == "\n":
            if len(sentence) > 0:
                sentences.append(" ".join(str(word) for word in sentence))
                sentences_labels.append(" ".join(str(label) for label in labels))
                sentence = []
                labels = []
            continue

        splits = line.split('\t')
        if splits[1] == '_':
            continue
        if splits[3] == 'satın':
            splits[3] = 'Noun'
        if splits[3] == 'Zero':
            splits[3] = 'Verb'

        sentence.append(splits[1])
        labels.append(splits[3])

    if len(sentence) > 0:
        sentences.append(" ".join(str(x) for x in sentence))
        sentences_labels.append(" ".join(str(x) for x in labels))

    return sentences, sentences_labels


def create_dataset_files(filename, dataset):
    with open(filename, 'w', encoding="utf-8") as file:
        for i in range(len(dataset)):
            sentences = dataset[i][0]
            labels = dataset[i][1]
            morphemes = dataset[i][2]
            for j in range(len(sentences)):
                file.write(sentences[j] + ' ' + labels[j] + ' ' + morphemes[j] + '\n')
            file.write('\n')


def prepare_data_sets(tag_problem_type, data_folder_path, filename, morph_code_folder, dataset_file):
    dataset = []
    analyzer = TurkishMorphemeAnalyzer(morph_code_folder)

    if dataset_file == 'conll':
        sentences, labels = read_conll_file(filename)
    elif dataset_file == 'enemax':
        sentences, labels = read_enemax_file(filename)
    else:
        raise ValueError('An unexpected dataset file type!')

    for i in range(len(sentences)):
        word_list = sentences[i].split(' ')
        label_list = labels[i].split(' ')
        morpheme_list = analyzer.get_sentence_morphemes(word_list)
        dataset.append([word_list, label_list, morpheme_list])

    jp.shutdownJVM()

    index1 = int(len(dataset)*0.8)
    index2 = int(len(dataset)*0.9)

    shuffle(dataset)
    create_dataset_files(data_folder_path + tag_problem_type + '_train.txt', dataset[:index1])
    create_dataset_files(data_folder_path + tag_problem_type + '_valid.txt', dataset[index1:index2])
    create_dataset_files(data_folder_path + tag_problem_type + '_test.txt', dataset[index2:])


if __name__ == "__main__":
    assert len(sys.argv) == 6

    problem_type = sys.argv[1]                      # ner or pos
    data_folder = sys.argv[2]                       # ../data/
    morph_folder = sys.argv[3]                      # ../
    dataset_path = data_folder + sys.argv[4]        # data_folder/file_name
    # pos filename = METUSABANCI_treebank_v-1.conll
    # ner filename = nerdata.txt || WFS7.MUClabeled
    dataset_type = sys.argv[5]                      # conll or enemax

    prepare_data_sets(problem_type, data_folder, dataset_path, morph_folder, dataset_type)
