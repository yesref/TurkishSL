
import numpy as np


def compute_f1(tag_type, predictions, correct, tag_alphabet):
    if tag_type not in ['NER', 'POS']:
        raise ValueError('An unexpected value for tag type!')

    label_pred = []
    for sentence in predictions:
        label_pred.append([tag_alphabet.get_instance(element) for element in sentence])

    label_correct = []
    for sentence in correct:
        label_correct.append([tag_alphabet.get_instance(element) for element in sentence])

    if tag_type == 'NER':
        prec = compute_precision(label_pred, label_correct)
        rec = compute_precision(label_correct, label_pred)
        f1 = 0
        if (rec+prec) > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        return prec, rec, f1
    else:
        acc = compute_accuracy(label_pred, label_correct)
        return acc


def compute_accuracy(guessed_sentences, correct_sentences):
    assert (len(guessed_sentences) == len(correct_sentences))
    correct_count = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = np.array(guessed_sentences[sentenceIdx])
        correct = np.array(correct_sentences[sentenceIdx])
        assert (len(guessed) == len(correct))

        count += guessed.size
        correct_count += sum(guessed == correct)

    accuracy = 0
    if count > 0:
        accuracy = float(correct_count) / count

    return accuracy


def compute_precision(guessed_sentences, correct_sentences):
    assert (len(guessed_sentences) == len(correct_sentences))
    correct_count = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert (len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B':  # A new chunk starts
                count += 1

                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctly_found = True

                    while idx < len(guessed) and guessed[idx][0] == 'I':  # Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctly_found = False

                        idx += 1

                    if idx < len(guessed):
                        if correct[idx][0] == 'I':  # The chunk in correct was longer
                            correctly_found = False

                    if correctly_found:
                        correct_count += 1
                else:
                    idx += 1
            else:
                idx += 1

    precision = 0
    if count > 0:
        precision = float(correct_count) / count

    return precision
