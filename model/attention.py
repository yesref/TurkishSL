# taken from https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2


from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, layers
from keras import backend as K
import tensorflow as tf


# taken from https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class VanillaAttention(Layer):
    def __init__(self, w_regularizer=None, u_regularizer=None, b_regularizer=None,
                 w_constraint=None, u_constraint=None, b_constraint=None, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(w_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(w_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.W, self.b, self.u = None, None, None

        super(VanillaAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init, name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer, constraint=self.W_constraint)
        self.b = self.add_weight((input_shape[-1],),
                                 initializer='zero', name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer, constraint=self.b_constraint)
        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init, name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer, constraint=self.u_constraint)

        super(VanillaAttention, self).build(input_shape)

    def call(self, x, mask=None):
        uit = dot_product(x, self.W) + self.b
        uit = K.tanh(uit)

        ait = dot_product(uit, self.u)
        a = K.exp(ait)

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        weighted_input = K.sum(x * a, axis=1)
        return weighted_input

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class SelfAttention(Layer):

    def __init__(self, output_dim, attention_width=None, attention_dropout=None,
                 q_init='glorot_uniform', q_constraint=None, q_regularize=None,
                 k_init='glorot_uniform', k_constraint=None, k_regularize=None,
                 v_init='glorot_uniform', v_constraint=None, v_regularize=None, **kwargs):
        self.output_dim = output_dim
        self.scale = self.output_dim ** 0.5
        self.attention_width = attention_width
        self.attention_dropout = attention_dropout
        self.WQ, self.WK, self.WV = None, None, None

        self.Q_initializer = initializers.get(q_init)
        self.K_initializer = initializers.get(k_init)
        self.V_initializer = initializers.get(v_init)

        self.Q_regularizer = regularizers.get(q_regularize)
        self.K_regularizer = regularizers.get(k_regularize)
        self.V_regularizer = regularizers.get(v_regularize)

        self.Q_constraint = constraints.get(q_constraint)
        self.K_constraint = constraints.get(k_constraint)
        self.V_constraint = constraints.get(v_constraint)

        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        batch_size, time_step, hidden_state = input_shape

        self.WQ = self.add_weight(shape=(hidden_state, self.output_dim),
                                  name='{}_WQ'.format(self.name),
                                  initializer=self.Q_initializer,
                                  regularizer=self.Q_regularizer,
                                  constraint=self.Q_constraint)

        self.WK = self.add_weight(shape=(hidden_state, self.output_dim),
                                  name='{}_WK'.format(self.name),
                                  initializer=self.K_initializer,
                                  regularizer=self.K_regularizer,
                                  constraint=self.K_constraint)

        self.WV = self.add_weight(shape=(hidden_state, self.output_dim),
                                  name='{}_WV'.format(self.name),
                                  initializer=self.V_initializer,
                                  regularizer=self.V_regularizer,
                                  constraint=self.V_constraint)

        super(SelfAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        queries = K.dot(inputs, self.WQ)
        keys = K.dot(inputs, self.WK)
        values = K.dot(inputs, self.WV)

        score = K.batch_dot(queries, keys, axes=[2, 2]) / self.scale

        time_step = K.shape(inputs)[1]
        if self.attention_width is not None and self.attention_width != 0:
            ones = tf.ones((time_step, time_step))
            local = tf.matrix_band_part(
                ones,
                K.minimum(time_step, self.attention_width // 2),
                K.minimum(time_step, (self.attention_width - 1) // 2),
            )
            score = score * K.expand_dims(local, 0)

        score = K.softmax(score)
        if self.attention_dropout is not None:
            score = layers.Dropout(self.attention_dropout)(score)

        output = K.batch_dot(score, values, axes=[2, 1])
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim


# taken from https://github.com/bojone/attention/blob/master/attention_keras.py
class MultiHeadSelfAttention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(MultiHeadSelfAttention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x, mask=None, **kwargs):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        else:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
