import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from sklearn.model_selection import train_test_split

#EMBEDDING_SIZE = 100
LSTM_UNITS = 200
DENSE_UNITS = 1
LEARNING_RATE = 0.001
BATCH_SIZE =10
# Gradient Clipping is IMPORTANT!!!!!!
CLIP_NORM = 1.0

MARGIN = 0.5
REG_LAMBDA = 0.00004
