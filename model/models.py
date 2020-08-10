from keras.models import Model
from keras.layers import *
from keras.initializers import RandomUniform
from keras_contrib.layers import CRF

from model.attention import VanillaAttention, SelfAttention
from model.elmo import ElmoEmbeddingLayer


def get_rnn(rnn_type, state_size, rnn_dropout, recurrent_dropout, sequences=True):
    if rnn_type == 'LSTM':
        return LSTM(state_size, return_sequences=sequences, dropout=rnn_dropout, recurrent_dropout=recurrent_dropout)
    elif rnn_type == 'GRU':
        return GRU(state_size, return_sequences=sequences, dropout=rnn_dropout, recurrent_dropout=recurrent_dropout)
    else:
        raise ValueError('An unexpected type of RNN!')


def self_attention(input_tensor, att_dim, att_width, dropout, masking=True, layer=1, is_residual=True):
    for _ in range(layer):
        output_tensor = SelfAttention(att_dim, att_width, dropout, masking)(input_tensor)
        if is_residual:
            input_tensor = concatenate([input_tensor, output_tensor])
            input_tensor = TimeDistributed(Dense(att_dim, activation='relu'))(input_tensor)
        else:
            input_tensor = output_tensor
    return input_tensor


class TaggingModel:
    def __init__(self, word_embeddings, max_word_length, max_morph_dim, morph_alphabet_size, char_alphabet_size,
                 label_alphabet, add_attention, use_morpheme, use_crf, use_cnn, rnn_type, character_dropouts,
                 rnn_dropout, recurrent_dropout, morph_dropouts, morph_state_size, use_morph_self_att, self_att_width,
                 self_att_dropout, self_att_concat_pos, char_emb_size=30, cnn_filter_size=30, cnn_win_size=3,
                 state_size=200):
        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        self.rnn_input = Embedding(input_dim=word_embeddings.shape[0], output_dim=word_embeddings.shape[1],
                                   weights=[word_embeddings], trainable=False)(words_input)
        #self.rnn_input2 = ElmoEmbeddingLayer()(words_input)
        self.model_input = [words_input]

        if use_morpheme:
            self._add_morpheme_features(add_attention, max_morph_dim, morph_alphabet_size, rnn_type, morph_state_size,
                                        morph_dropouts, use_morph_self_att, self_att_width, self_att_dropout,
                                        self_att_concat_pos)
        if use_cnn:
            self._add_character_features(max_word_length, char_alphabet_size, char_emb_size, character_dropouts,
                                         cnn_win_size, cnn_filter_size)

        rnn_out = Bidirectional(get_rnn(rnn_type, state_size, rnn_dropout, recurrent_dropout))(self.rnn_input)
        # out = MultiHeadSelfAttention(2, int(state_size/2))([rnn_out, rnn_out, rnn_out, rnn_out, rnn_out])

        rnn_out = self_attention(rnn_out, state_size*2, self_att_width, self_att_dropout, masking=True)

        if use_morpheme and use_morph_self_att and self_att_concat_pos == 'BeforeCRF':
            rnn_out = concatenate([rnn_out, self.self_att_output])

        if use_crf:
            crf = CRF(label_alphabet.size())
            output = crf(rnn_out)
            self.default_loss = crf.loss_function
            self.accuracy = [crf.accuracy]
        else:
            output = TimeDistributed(Dense(label_alphabet.size(), activation='softmax'))(rnn_out)
            self.default_loss = 'categorical_crossentropy'
            self.accuracy = None

        self.model = Model(inputs=self.model_input, outputs=[output])

    def get_model(self):
        return self.model

    def set_model(self, loaded_model):
        self.model = loaded_model

    def _add_morpheme_features(self, add_attention, max_morph_dim, morph_alphabet_size, rnn_type, rnn_state_size,
                               morph_dropout, use_self_att, self_att_width, self_att_dropout, self_att_concat_pos,
                               self_att_masking=True, self_att_conn=True, self_att_stack=1):
        # morpheme model with single rnn and attention for morphemes of a word
        morph_inp = Input(shape=(max_morph_dim, morph_alphabet_size,), name='morph_inputs')
        if add_attention:
            morph_out = Bidirectional(get_rnn(rnn_type, rnn_state_size, morph_dropout, morph_dropout))(morph_inp)
            morph_out = VanillaAttention()(morph_out)
        else:
            morph_rnn = get_rnn(rnn_type, rnn_state_size, morph_dropout, morph_dropout, sequences=False)
            morph_out = Bidirectional(morph_rnn)(morph_inp)
        morph_layer = Model(morph_inp, morph_out)

        # we used TimeDistributed to provide our model use newly created morpheme model for each word
        word_morph_input = Input(shape=(None, max_morph_dim, morph_alphabet_size), name='word_morph_inputs')
        word_morph_output = TimeDistributed(morph_layer)(word_morph_input)

        self.model_input = self.model_input + [word_morph_input]
        if use_self_att:
            self.self_att_output = self_attention(word_morph_output, rnn_state_size*2, self_att_width, self_att_dropout,
                                                  self_att_masking, layer=self_att_stack, is_residual=self_att_conn)
        if use_self_att and self_att_concat_pos == 'BeforeRNN':
            self.rnn_input = concatenate([self.rnn_input, self.self_att_output])
        else:
            self.rnn_input = concatenate([self.rnn_input, word_morph_output])

    def _add_character_features(self, max_word_length, char_alphabet_size, embed_dim, dropouts, win_size, filter_size):
        character_input = Input(shape=(None, max_word_length,), name='char_input')
        embedded_chars = TimeDistributed(Embedding(
            input_dim=char_alphabet_size, output_dim=embed_dim,
            embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
            name='char_embedding')(character_input)
        embedded_chars = Dropout(dropouts[0])(embedded_chars)

        cnn_out = TimeDistributed(Conv1D(kernel_size=win_size, filters=filter_size, padding='same',
                                         activation='tanh', strides=1))(embedded_chars)
        cnn_out = TimeDistributed(MaxPooling1D(max_word_length))(cnn_out)
        cnn_out = TimeDistributed(Flatten())(cnn_out)
        characters = Dropout(dropouts[1])(cnn_out)

        self.model_input = self.model_input + [character_input]
        self.rnn_input = concatenate([self.rnn_input, characters])

    def compile(self, optimizer, loss=None):
        model_loss = loss if loss is not None else self.default_loss
        self.model.compile(loss=model_loss, optimizer=optimizer, metrics=self.accuracy)

    def train(self, use_cnn, use_morpheme, words, chars, morphemes, labels):
        model_input = [words, morphemes] if use_morpheme else [words]
        model_input = model_input + [chars] if use_cnn else model_input

        return self.model.train_on_batch(model_input, np.array(labels))

    def validate(self, use_cnn, use_morpheme, words, chars, morphemes, labels):
        model_input = [words, morphemes] if use_morpheme else [words]
        model_input = model_input + [chars] if use_cnn else model_input

        return self.model.test_on_batch(model_input, np.array(labels))

    def predict(self, use_cnn, use_morpheme, words, chars, morphemes):
        model_input = [words, morphemes] if use_morpheme else [words]
        model_input = model_input + [chars] if use_cnn else model_input

        return self.model.predict(model_input)[0].argmax(axis=-1)

