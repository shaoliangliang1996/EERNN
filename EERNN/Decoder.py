from hyper_params import *

def choose_lstm(lstm_units):

    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(
            name="lstm",
            units=lstm_units,
            return_sequences=True,
            return_state=False,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1549193338, maxval=0.1549193338),
            # kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        )
    else:
        return tf.keras.layers.LSTM(
            name='lstm',
            units=lstm_units,
            return_sequences=True,
        return_state = False,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1549193338, maxval=0.1549193338),
            # kernel_regularizer=tf.keras.regularizers.l2(REG_LAMDA)
        )
def choose_bilstm(LSTM_UNITS):

    if tf.test.is_gpu_available():
        return tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(
            name="bi_lstm",
            units=LSTM_UNITS,
            return_sequences=True,
            return_state=False,
            #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-tf.rsqrt(6 / (100 + 50)),
            #                                                       maxval=tf.rsqrt(6 / (100 + 50))),
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.2,
                                                                   maxval=0.2),
            kernel_regularizer=tf.keras.regularizers.l2(0.00004),
        ))
    else:
        return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            name="bi_lstm",
            units=LSTM_UNITS,
            return_sequences=True,
            return_state=False,
            #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-tf.rsqrt(6 / (100 + 50)),
            #                                                       maxval=tf.rsqrt(6 / (100 + 50))),
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.2,
                                                                   maxval=0.2),
            kernel_regularizer=tf.keras.regularizers.l2(0.00004),
        ))



class Decoder(tf.keras.Model):
    def __init__(self,embedding_matrix,embedding_matrix2):
        super(Decoder, self).__init__()
        self.lstm =choose_lstm(LSTM_UNITS)

        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense1 = tf.keras.layers.Dense(
            name='dense1',
            units=50,
            activation=tf.keras.activations.relu,
            #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-tf.rsqrt(6/(150+50)), maxval=tf.rsqrt(6/(150+50))),
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1732050808,
                                                                   maxval=0.1732050808),
            #kernel_regularizer=tf.keras.regularizers.l2(0.00004)
        )
        self.dense2 = tf.keras.layers.Dense(
            name='dense2',
            units=1,
            #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-tf.rsqrt(6/(1+50)),maxval=tf.rsqrt(6/(1+50))),
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.3429971703,maxval=0.3429971703),
            #kernel_regularizer=tf.keras.regularizers.l2(0.00004)
        )
        self.sotfmax = tf.keras.layers.Softmax()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=NUM_WORDS,
            output_dim=EMBEDDING_DIM,
            name="embedding",
            # embeddings_initializer=tf.keras.initializers.RandomUniform(),
            # input_length =
            weights=[embedding_matrix],
            trainable=False,
        )
        self.embedding2 = tf.keras.layers.Embedding(
            input_dim=2,
            output_dim=4*LSTM_UNITS,
            name="embedding2",
            # embeddings_initializer=tf.keras.initializers.RandomUniform(),
            # input_length =
            weights=[embedding_matrix2],
            trainable=False,
        )
        self.bi_lstm = choose_bilstm(LSTM_UNITS)

    def call_encode(self, pro_dic):
        x = self.embedding(pro_dic)
        if pro_dic.shape[0]<2000:
            x = self.bi_lstm(x)
        else:
            x1 = self.bi_lstm(x[0:2000, :, :])
            x2 = self.bi_lstm(x[2000:4000, :, :])
            x3 = self.bi_lstm(x[4000: ,:, :])
            x = tf.concat([x1, x2, x3], axis=0)
        x = tf.reduce_max(x, axis=1, keep_dims=False, name=None)
        cos_X = self.cosine(x, x)
        #x (num_problem,2*LSTM_UNITS)
        #cosx (num_problem,num_problem)
        return x, cos_X

    def cosine(self, q, a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(q), 1))#按行求和
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(a), 1))
        #pre_pooled_len_1,,pooled_len_2  (num_problem,)
        #print('pre_pooled_len_1,,pooled_len_2', pooled_len_1.shape, pooled_len_2.shape)
        pooled_len_1 = tf.expand_dims(pooled_len_1, axis=0)
        pooled_len_2 = tf.expand_dims(pooled_len_2, axis=-1)
        #pooled_len_1,,pooled_len_2  (1,num_problem)(num_problem,1)
        norm_matrix = tf.tensordot(pooled_len_2, pooled_len_1, [[1], [0]])
        dot_matrix = tf.tensordot(a, q, [[1], [1]])
        #norm_matrix,dot_matrix （num_problem,num_problem）
        sim = dot_matrix / norm_matrix
        # sim == (batch_size num_pro)
        return sim

    def call(self, data,num_pro,X,cos_X,trimatrix):
        data_target, data_cor = data
        data_target_one_hot = tf.one_hot(data_target, num_pro)
        #data_target_one_hot (batch_size,max_step,num_problem)
        data_cor_embedding = self.embedding2(data_cor)
        #data_cor_embedding (batch_size,max_step,4* LSTM_UNITS)
        t_X = tf.tensordot(data_target_one_hot, X, [[2], [0]])
        #print('t_X',t_X.shape) batch_size,max_step,2* LSTM_UNITS)
        t_X = tf.concat([t_X, t_X], axis=2)
        xt = tf.multiply(t_X, data_cor_embedding)
        # xt == shape(batch_size, 1,4 * LSTM_UNITS)

        ht = self.lstm(xt)
        hatt= self.cal_hatt(ht,data_target_one_hot,X,cos_X,trimatrix)
        r = self.dense1(hatt)
        r = self.dense2(r)
        return r

    def cal_hatt(self,ht,data_target_one_hot,X,cos_X,trimatrix):
        a = tf.tensordot(data_target_one_hot,cos_X,[[2],[0]])
        #a (batch_size,max_step,num_problem)
        aj = tf.expand_dims(a,-1)
        hj = tf.expand_dims(ht,2)
        hidden = tf.multiply(aj,hj)
        #hidden （batch_size,max_step，num_problem，LSTM_UNITS)
        #trimatrix 上三角（max_step，max_step）
        ajhj = tf.tensordot(hidden,trimatrix,[[1],[0]])
        #(BATCH_SIZE,numpro,unitstate,maxstep)
        ajhj = tf.transpose(ajhj, [0, 3, 1,2])
        #(BATCH_SIZE,maxstep,numpro,unitstate)
        x1 = tf.expand_dims(X, 0)
        x1 = tf.expand_dims(x1, 0)
        XX = tf.tile(x1, multiples=[BATCH_SIZE,data_target_one_hot.shape[1], 1, 1])
        hatt = tf.concat([ajhj , XX],-1)
        #print('hatt',hatt.shape)#(batchsize,maxstep,numpro，3*LSTM_UNITS)
        return hatt


