import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os



NUM_WORDS = 5000
BUFFER_SIZE = 12
BATCH_SIZE = 32
EMBEDDING_DIM = 50
LSTM_UNITS = 50
PREFETCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 5120
MAXLEN = 100

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