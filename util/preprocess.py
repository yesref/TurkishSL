
from keras.preprocessing.sequence import pad_sequences

import numpy as np


def get_data_ids(alphabets, sentences, labels, morphemes, max_morph_dim):
    word_alphabet, char_alphabet, label_alphabet, morph_alphabet = alphabets

    word_ids = [[word_alphabet.get_index(word) for word in sentence] for sentence in sentences]
    label_ids = [[label_alphabet.get_index(label) for label in stc_labels] for stc_labels in labels]

    max_word_length = len(max(word_alphabet.get_content()['instances'], key=len))
    char_ids = [[[char_alphabet.get_index(char) for char in word] for word in sentence] for sentence in sentences]
    padded_char_ids = [pad_sequences(chars, max_word_length, padding='post') for chars in char_ids]

    morph_ids = [[[morph_alphabet.get_index(m) for m in morphs] for morphs in stc_morphs] for stc_morphs in morphemes]
    padded_morph_ids = [pad_sequences(morphs, max_morph_dim, padding='post') for morphs in morph_ids]

    dataset = [[word_ids[i], padded_char_ids[i], padded_morph_ids[i], label_ids[i]] for i in range(len(word_ids))]

    return dataset


def get_batch_lengths(dataset):
    dataset.sort(key=lambda d: len(d[1]))

    batch_length = len(dataset[0][0])
    batch_lengths = []

    for i in range(len(dataset)):
        length = len(dataset[i][0])
        if length > batch_length:
            batch_lengths.append(i)
            batch_length = length

    return dataset, batch_lengths


def iterate_batches(dataset, batch_lengths):

    start = 0

    for batch_length in batch_lengths:
        batch = dataset[start:batch_length]
        start = batch_length
        batch_t = [[stc[i] for stc in batch] for i in range(4)]
        words = np.asarray(batch_t[0])
        chars = np.asarray(batch_t[1])
        morphs = np.asarray(batch_t[2])
        labels = batch_t[3]  # labels = np.asarray(np.expand_dims(labels,-1)

        yield words, chars, morphs, labels

    return
