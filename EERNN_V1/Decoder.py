from hyper_params import *

def choose_lstm(lstm_units):

    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(
            name="lstm",
            units=lstm_units,
            return_sequences=True,
            return_state=False,
            # kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),

        )
    else:
        return tf.keras.layers.LSTM(
            name='lstm',
            units=lstm_units,
            return_sequences=True,
        return_state = False,
            dropout=0.1,
            # kernel_regularizer=tf.keras.regularizers.l2(REG_LAMDA)
        )
class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = choose_lstm(LSTM_UNITS)
        self.dropout = tf.keras.layers.Dropout(0, 1)
        self.dense = tf.keras.layers.Dense(
            name='dense',
            units=1,
            activation=tf.keras.activations.relu,
            # kernel_regularizer=tf.keras.regularizers.l2(REG_LAMDA)
        )
        self.sotfmax = tf.keras.layers.Softmax()

    def call(self, xt,data,hiden,X,cos_X):
        ht = self.lstm(xt)
        #print('ht', ht.shape)
        #ht = (batch_size,1,LSTM_UNITS)
        hatt ,hiden= self.cal_hatt(ht,data,hiden,X,cos_X)
        #print('hatt', hatt.shape)
        r = self.dense(hatt)
        ## r = self.sotfmax(r)
        return r,hiden

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

    def cal_hatt(self,ht,data,hiden,X,cos_X):
        def _compute_ajhj(h):
            aj,ht = h
            return tf.tensordot(aj,ht,[[1],[0]])
        target_id = tf.slice(data, [0, 0, 0], [data.shape[0], 1, 1])
        #print('tartget_id ',target_id.shape)
        target_id = tf.reshape(target_id,[-1])########
        #target_id= tf.map_fn(lambda i: BATCH_SIZE * i + flat_target_id[i], range(len(flat_target_id)))
        a = tf.gather(cos_X,target_id)
        aj = tf.expand_dims(a,axis=-1)
        #print('a ',a.shape)
        #aj = tf.expand_dims(self.cosine(X,a),axis=-1)
        #ajhj = tf.tensordot(aj,ht,[[2],[]])(32,434,1)(32,1,50)  只求后两者 不知道怎么用这个求 有个对于的batch_size
        hidden = tf.map_fn( _compute_ajhj, elems=(aj, ht), dtype=tf.float32)
        #hidden = tf.map_fn(lambda h:tf.tensordot(h[0],h[1],[[1],[0]]),(aj,ht))
        #print('hidden',hidden.shape)
        ajhj = tf.add(hidden,hiden)
        #print('ajhj', ajhj.shape)
        #hatt = tf.map_fn(lambda aa:tf.concat([aa,X],1),ajhj)
        XX = tf.tile(tf.expand_dims(X, 0), multiples=[BATCH_SIZE, 1, 1])
        hatt = tf.concat([ajhj , XX],-1)
        return hatt,hidden


