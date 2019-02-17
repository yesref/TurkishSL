
from util.logger import get_logger


def read_embeddings_file(word_alphabet, filename):
    logger = get_logger('Embedding')
    logger.info("Reading embedding from %s" % filename)

    word_contents = word_alphabet.get_content()['instances']
    lower_word_contents = [word.lower() for word in word_contents]

    file = open(filename, encoding="utf8")

    embeddings = dict()
    embeddings_size = 0

    for line in file:
        tokens = line.strip().split()

        # ignore vector size and dimension at the beginning lines if exist
        if len(tokens) < 3:
            continue

        # get embedding dimension
        if embeddings_size == 0:
            embeddings_size = len(tokens)-1

        # all the lines must have same dimensionality
        assert embeddings_size == len(tokens)-1

        embeddings[tokens[0]] = tokens[1:]

    file.close()

    logger.info("Embeddings vector read from file.")
    logger.info("Number of words in embeddings: %d" % len(embeddings))
    logger.info("Dimension of embedding vectors : %d" % embeddings_size)

    return embeddings, embeddings_size
