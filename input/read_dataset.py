
from util.alphabet import Alphabet
from util.logger import get_logger


def read_file(filename, word_alphabet, char_alphabet, tag_alphabet, morph_alphabet):
    file = open(filename, encoding="utf8")

    words, labels, morphemes = [], [], []
    sentences, sentence_labels, sentence_morphemes = [], [], []

    for line in file:

        if len(line) == 0 or line[0] == "\n":
            if len(words) > 0:
                sentences.append(words)
                sentence_labels.append(labels)
                sentence_morphemes.append(morphemes)
                words, labels, morphemes = [], [], []
            continue

        tokens = line.split()

        word_alphabet.add(tokens[0])
        for char in tokens[0]:
            char_alphabet.add(char)
        tag_alphabet.add(tokens[1])

        word_morphemes = []
        for morpheme in tokens[2].strip().split(','):
            morph_alphabet.add(morpheme)
            word_morphemes.append(morpheme)

        words.append(tokens[0])
        labels.append(tokens[1])
        morphemes.append(word_morphemes)

    return sentences, sentence_labels, sentence_morphemes


def read_data_files(train_file_name, dev_file_name=None, test_file_name=None):
    logger = get_logger("Read Datasets")

    word_alphabet = Alphabet('word', default_value=True, singleton=True)
    char_alphabet = Alphabet('character', default_value=True)
    tag_alphabet = Alphabet('tag')
    morph_alphabet = Alphabet('morpheme')
    logger.info('Word, Char, Tag and Morpheme Alphabets will be created')

    word_alphabet.add('_PAD')
    char_alphabet.add('_PAD_CHAR')
    tag_alphabet.add('_PAD_TAG')
    morph_alphabet.add('_PAD_MORPH')

    train_sentences, train_labels, train_morphemes = \
        read_file(train_file_name, word_alphabet, char_alphabet, tag_alphabet, morph_alphabet)
    logger.info("Training data was read. Number of sentences: %d" % len(train_sentences))

    dev_sentences, dev_labels, dev_morphemes, test_sentences, test_labels, test_morphemes = [], [], [], [], [], []
    if dev_file_name is not None:
        dev_sentences, dev_labels, dev_morphemes = \
            read_file(dev_file_name, word_alphabet, char_alphabet, tag_alphabet, morph_alphabet)
        logger.info("Cross validation (dev) data was read. Number of sentences: %d" % len(dev_sentences))
    if test_file_name is not None:
        test_sentences, test_labels, test_morphemes = \
            read_file(test_file_name, word_alphabet, char_alphabet, tag_alphabet, morph_alphabet)
        logger.info("Test data was read. Number of sentences: %d" % len(test_sentences))

    word_alphabet.close()
    char_alphabet.close()
    tag_alphabet.close()
    morph_alphabet.close()

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("Tag Alphabet Size: %d" % tag_alphabet.size())
    logger.info("Morpheme Alphabet Size: %d" % morph_alphabet.size())

    return (word_alphabet, char_alphabet, tag_alphabet, morph_alphabet), (train_labels, dev_labels, test_labels), \
           (train_sentences, dev_sentences, test_sentences), (train_morphemes, dev_morphemes, test_morphemes)
