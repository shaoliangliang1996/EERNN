from train_exercise import *
from hyper_params import *
from Decoder import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import time
import numpy as np
from  EERNNDataProcessor import EERNNDataProcessor
import os

def Evaluate(DataName,TmpDir):
    trimatrix = np.tri(MAXLEN, MAXLEN, 0).T
    # trimatrix = tf.reshape(trimatrix,shape=[MAXLEN,MAXLEN])
    trimatrix = tf.cast(trimatrix, tf.float32)
    DataProssor = EERNNDataProcessor([15, 1000000, 0.06, 1], [10, 1000000, 0.02, 1],
                                     ['2005-01-01 23:47:31', '2019-01-02 11:21:49'], True, DataName, TmpDir)
    pro_dic, embedding_matrix, dataset, test_data, embedding_matrix2 = DataProssor.LoadEERNNData(BATCH_SIZE,
                                                                                                 PREFETCH_SIZE,
                                                                                                 SHUFFLE_BUFFER_SIZE,
                                                                                                 LSTM_UNITS, 100)
    evaluate(0, test_data, embedding_matrix, embedding_matrix2, pro_dic, trimatrix)

if __name__ == "__main__":
    Evaluate('hdu',"/media/data6t/educationData/submitData/shao/Data/datapart/")