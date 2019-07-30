from hyper_params import *

def choose_lstm(lstm_units):

    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(
            name="lstm",
            units=lstm_units,
            return_sequences=True,
            return_state=False,
            # kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
            # dropout=keep_prob,#这里是这么设的么
        )
    else:
        return tf.keras.layers.LSTM(
            name='lstm',
            units=lstm_units,
            return_sequences=True,
        return_state = False,
            # dropout=keep_prob,#正则和dropout冲突么 Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
            # kernel_regularizer=tf.keras.regularizers.l2(REG_LAMDA)
        )

class DKTModel(tf.keras.Model):
    def __init__(self,config,training=True):
        super(DKTModel,self).__init__(name='dkt_model')
        self.lstm = choose_lstm(config['hidden_neurons'])
        self.dropout = tf.keras.layers.Dropout(rate=config['keep_prob'])
        self.dense=tf.keras.layers.Dense(
            name='dense',
            units=config['dense_unit'],
            activation=tf.keras.activations.sigmoid,
            #kernel_regularizer=tf.keras.regularizers.l2(REG_LAMDA)
            )
    def call(self,x):
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x

