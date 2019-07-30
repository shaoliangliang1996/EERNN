from hyper_params import *

class Encoder(tf.keras.Model):
    def __init__(self,embedding_matrix):
        super(Encoder,self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=NUM_WORDS,
            output_dim=EMBEDDING_DIM,
            embeddings_initializer=tf.keras.initializers.RandomUniform(),
            # input_length =
            weights=[embedding_matrix],
            trainable=False,
        )
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            name="lstm",
            units=LSTM_UNITS,
            return_sequences=True,
            return_state=False,
            # kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        ))

    def call(self,prodata):
        x = self.embedding(prodata)
        x = self.bi_lstm(x)
        x = tf.reduce_max(x, axis=1, keep_dims=False, name=None)
        #cos_X = self.cosine(x,x)
        #print('cos_X',cos_X.shape)
        return x
    def cosine(self,q, a):
        #print('q,a',q.shape,a.shape)
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(q),1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(a),1))
        pooled_len_1 = tf.expand_dims(pooled_len_1,axis=0)
        pooled_len_2 = tf.expand_dims(pooled_len_2,axis=-1)
        #print('pooled_len_1,,pooled_len_2',pooled_len_1.shape,pooled_len_2.shape)
        norm_matrix = tf.tensordot(pooled_len_2,pooled_len_1, [[1], [0]])
        dot_matrix = tf.tensordot(a, q, [[1], [1]])
        #print('norm_matrix,dot_matrix',norm_matrix.shape,dot_matrix.shape)
        sim = dot_matrix / norm_matrix
        #sim == (batch_size num_pro)
        return sim
