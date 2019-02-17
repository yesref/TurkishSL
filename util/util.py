

def get_max_word_morpheme_dim(train_stc, dev_stc, test_stc):
    max_dim = len(max([max(morph, key=len) for morph in train_stc], key=len))
    max_dim = max(max_dim, len(max([max(morph, key=len) for morph in dev_stc], key=len)))
    max_dim = max(max_dim, len(max([max(morph, key=len) for morph in test_stc], key=len)))

    return max_dim
