from OJDataProcessor import OJDataProcessor
from OJProblemtextProcessor import OJProblemtextProcessor
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

class EERNNDataProcessor(object):
    def __init__(self,userLC,problemLC,timeLC,OnlyRight,DataName = 'hdu',TmpDir = '/home/shaoliangliang/code/MANN_TCA_SA/datapart/'):
        self.OJData = OJDataProcessor(DataName,TmpDir)
        self.OJData.LoadSubmitRecordData(userLC,problemLC,timeLC,OnlyRight)
        self.OJProblem = OJProblemtextProcessor(userLC,problemLC,timeLC,OnlyRight,DataName,TmpDir)

    def LoadEERNNData(self,BATCH_SIZE,PREFETCH_SIZE,SHUFFLE_BUFFER_SIZE,LSTM_UNITS, maxLen=100):
        np_sample = len(self.OJData.submitRecord)
        seqs_ans = {}
        for i in range(len(self.OJData.submitRecord)):
            if (len(self.OJData.submitRecord[i]) <= maxLen):
                seqs_ans[i] = self.OJData.submitRecord[i]
            else:
                nowId = len(self.OJData.submitRecord[i])
                seqs_ans[i] = self.OJData.submitRecord[i][nowId - maxLen:nowId]
                nowId -= maxLen
                while (nowId >= 0):
                    if (nowId - maxLen >= 0):
                        seqs_ans[np_sample] =self.OJData.submitRecord[i][nowId - maxLen:nowId]
                    else:
                        seqs_ans[np_sample] = self.OJData.submitRecord[i][0:maxLen]
                    np_sample += 1
                    nowId -= maxLen
        pro_dic,cor_dic = [],[]
        for key in seqs_ans.keys():
            pro ,cor= [],[]
            for i in range(len(seqs_ans[key])):
                pro.append(seqs_ans[key][i][0])
                cor.append(seqs_ans[key][i][1])
            pro_dic.append(pro)
            cor_dic.append(cor)
        tmp_pro = tf.keras.preprocessing.sequence.pad_sequences(pro_dic, value=0, padding='post', truncating='post',maxlen=maxLen)
        tmp_cor = tf.keras.preprocessing.sequence.pad_sequences(cor_dic, value=0, padding='post', truncating='post', maxlen=maxLen)
        data = np.concatenate((tmp_pro, tmp_cor), axis=1)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)
        print('train_data', train_data.shape)
        dataset_train_pro = tf.data.Dataset.from_tensor_slices(train_data[:, :maxLen])
        dataset_train_cor = tf.data.Dataset.from_tensor_slices(train_data[:, maxLen:])
        dataset_train = tf.data.Dataset.zip((dataset_train_pro, dataset_train_cor))

        dataset_test_pro = tf.data.Dataset.from_tensor_slices(test_data[:, :maxLen])
        dataset_test_cor = tf.data.Dataset.from_tensor_slices(test_data[:, maxLen:])
        dataset_test = tf.data.Dataset.zip((dataset_test_pro, dataset_test_cor))

        dataset_train = dataset_train.shuffle(buffer_size=tf.constant(SHUFFLE_BUFFER_SIZE, dtype=tf.int64)).batch(BATCH_SIZE, drop_remainder=True)
        # .repeat(1).prefetch(PREFETCH_SIZE).
        dataset_test = dataset_test.prefetch(PREFETCH_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        onehot_matrix = np.zeros((2, 4 * LSTM_UNITS))
        onehot_matrix[0] = [0] * 2 * LSTM_UNITS + [1] * 2 * LSTM_UNITS
        onehot_matrix[1] = [1] * 2 * LSTM_UNITS + [0] * 2 * LSTM_UNITS

        pro_dic, embedding_matrix = self.OJProblem.Problem2Tensor()
        return  pro_dic, embedding_matrix,dataset_train, dataset_test, onehot_matrix
