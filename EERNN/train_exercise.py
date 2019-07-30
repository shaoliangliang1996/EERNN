from hyper_params import *
from Decoder import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import time
import numpy as np
from  EERNNDataProcessor import EERNNDataProcessor
import os

def cal_flat_target_logits(prediction,target_id,target_correctness):
    num_pro = prediction.shape[-2]
    prediction,target_id, target_correctness = prediction[:,:-1,:,:],target_id[:,1:],target_correctness[:,1:]
    flat_logits = tf.reshape(prediction, [-1])
    flat_target_correctness = tf.reshape(target_correctness, [-1])
    flat_bias_target_id = num_pro * tf.range(BATCH_SIZE * target_id.shape[-1])
    flat_target_id = tf.reshape(target_id, [-1])+flat_bias_target_id
    flat_target_logits = tf.gather(flat_logits, flat_target_id)  ########
    return flat_target_logits,flat_target_correctness

def entroy_loss(prediction, data):
    target_id, target_correctness = data
    flat_target_logits, flat_target_correctness = cal_flat_target_logits(prediction,target_id,target_correctness)
    flat_target_correctness = tf.cast(flat_target_correctness,dtype=tf.float32)
    return tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness, logits=flat_target_logits))
def cal_pre(prediction,data):
    target_id, target_correctness = data
    flat_target_logits, flat_target_correctness = cal_flat_target_logits(prediction, target_id,target_correctness)
    pred = tf.sigmoid(flat_target_logits)
    binary_pred = tf.cast(tf.greater_equal(pred, 0.5), tf.int32)
    return pred, binary_pred,flat_target_correctness

def run(DataName,TmpDir):
    trimatrix = np.tri(MAXLEN, MAXLEN, 0).T
    # trimatrix = tf.reshape(trimatrix,shape=[MAXLEN,MAXLEN])
    trimatrix = tf.cast(trimatrix, tf.float32)
    DataProssor = EERNNDataProcessor([15,1000000,0.06,1],[10,1000000,0.02,1],['2005-01-01 23:47:31','2019-01-02 11:21:49'],True,DataName,TmpDir)
    pro_dic, embedding_matrix, dataset,test_data, embedding_matrix2 = DataProssor.LoadEERNNData(BATCH_SIZE, PREFETCH_SIZE, SHUFFLE_BUFFER_SIZE, LSTM_UNITS,100)
    print("Start training...")
    epochs = 10
    decoder = Decoder(embedding_matrix,embedding_matrix2)
    lr = 0.01
    lr_decay = 0.92
    for epoch in range(0,epochs):
        optimizer = tf.train.AdamOptimizer(lr * lr_decay ** epoch)
        start = time.time()
        total_loss = 0
        dataset.shuffle(BUFFER_SIZE)
        for batch, (data) in enumerate(dataset):
            data_target,data_cor = data
            loss = 0
            with tf.GradientTape() as tape:
                X,cos_X =  decoder.call_encode(pro_dic)
                prediction = decoder.call(data,pro_dic.shape[0],X,cos_X,trimatrix)
                #data_t == (batch_size,1,2)
                loss += entroy_loss(prediction, data)
            batch_loss = (loss/int(data_target.shape[1]))
            total_loss += batch_loss
            variables =decoder.lstm.variables+decoder.dense2.variables+decoder.dense1.variables+decoder.bi_lstm.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            print('.', end='')
            if batch%100 == 0:
                print()
                print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy())
                      )
        end = time.time()
        decoder.save_weights('/media/data6t/educationData/submitData/shao/EERNN/model2/my_model_'+str(epoch+1))
        print("Epoch {} cost {}".format(epoch + 1, end - start))
        evaluate(epoch, test_data, embedding_matrix, embedding_matrix2, pro_dic, trimatrix)

def evaluate(epoch,test_data,embedding_matrix, embedding_matrix2,pro_dic,trimatrix):
    decoder = Decoder(embedding_matrix, embedding_matrix2)
    decoder.load_weights('/media/data6t/educationData/submitData/shao/EERNN/model2/my_model_' + str(epoch + 1))
    print("loadWeight!!!")
    f = open('/media/data6t/educationData/submitData/shao/EERNN/data/acc.txt', 'w')
    print('start evaluation')
    preds, targets, binary_preds = [], [], []
    X, cos_X = decoder.call_encode(pro_dic)
    for batch, (data) in enumerate(test_data):
        prediction = decoder.call(data, pro_dic.shape[0], X, cos_X, trimatrix)
        # data_t == (batch_size,1,2)
        pred, binary_pred, target_correctness = cal_pre(prediction, data)

        preds.append(pred)
        binary_preds.append(binary_pred)
        targets.append(target_correctness)

    preds = np.concatenate(preds)
    binary_preds = np.concatenate(binary_preds)
    targets = np.concatenate(targets)

    np.savetxt("/media/data6t/educationData/submitData/shao/EERNN/result2/" + str(epoch + 1) + "preds.txt", preds)
    np.savetxt("/media/data6t/educationData/submitData/shao/EERNN/result2/" + str(epoch + 1) + "targets.txt", targets)
    np.savetxt("/media/data6t/educationData/submitData/shao/EERNN/result2/" + str(epoch + 1) + "binary_preds.txt",binary_preds)

    # auc_value = roc_auc_score(targets,preds)
    accuracy = accuracy_score(targets, binary_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
    f.write(str(epoch + 1) + ", auc={0}, accuracy={1}, precision={2}, recall={3} \n".format(0, accuracy,
                                                                                            precision, recall))
    print("\n auc={0}, accuracy={1}, precision={2}, recall={3}".format(0, accuracy, precision, recall))
    f.close()

if __name__=='__main__':
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.keras.backend.set_session(tf.Session(config=config))
    run('hdu',"/media/data6t/educationData/submitData/shao/Data/datapart/")
