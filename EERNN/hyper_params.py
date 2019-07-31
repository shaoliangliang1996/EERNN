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
MAXLEN = 10
EPOCHS=10
LR = 0.01
LR_DECAY = 0.92