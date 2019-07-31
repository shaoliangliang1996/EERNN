from hyper_params import *
from execise_data import *
from student_embed import *
from Encoder import *
from Decoder import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import time
import os

def cal_flat_target_logits(prediction,data_t):
    num_pro = prediction.shape[-1]
    target_id = tf.slice(data_t, [0, 0, 0], [data_t.shape[0], 1, 1])
    target_correctness = tf.slice(data_t, [0, 0, 1], [data_t.shape[0], 1, 1])
    flat_logits = tf.reshape(prediction, [-1])
    flat_target_correctness = tf.reshape(target_correctness, [-1])
    flat_target_id = tf.reshape(target_id, [-1])
    flat_target_id = flat_target_id + num_pro * tf.constant(range(flat_target_id.shape[0]))
    flat_target_logits = tf.gather(flat_logits, flat_target_id)  ########
    return flat_target_logits,flat_target_correctness

def entroy_loss(prediction, data_t):
    flat_target_logits, flat_target_correctness = cal_flat_target_logits(prediction,data_t)
    flat_target_correctness = tf.cast(flat_target_correctness,dtype=tf.float32)
    return tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness, logits=flat_target_logits))
def cal_pre(prediction, data_t):
    flat_target_logits, flat_target_correctness = cal_flat_target_logits(prediction, data_t)
    pred = tf.sigmoid(flat_target_logits)
    binary_pred = tf.cast(tf.greater_equal(pred, 0.5), tf.int32)
    return pred, binary_pred,flat_target_correctness


def data_forstu(data, X):#这个位置加循环应该没有问题吧  做这一步这能循环了
    z = tf.zeros(shape=[1, 2 * LSTM_UNITS], dtype=tf.float32)
    xt =tf.zeros(shape=[1,4*LSTM_UNITS],dtype=tf.float32)  # ([BATCH_SIZE,data.shape[1],4*LSTM_UNITS])
    for i in data:
        for j in i:
            xi = tf.reshape(X[j[0]],shape=[1, 2*LSTM_UNITS])
            if j[1] == 1:
                xt = tf.concat([xt,tf.concat([xi,z],1)],0)
            else:
                xt = tf.concat([xt,tf.concat([z,xi], 1)], 0)
    xt = xt[1:,:]
    return xt

def run(dataset,test_data,pro_dic,embedding_matrix,prodata):
    f = open('/media/data6t/educationData/submitData/shao/EERNN/data/acc.txt', 'w')
    print("Start training...")
    epochs = 10
    optimizer = tf.train.AdamOptimizer(0.001)
    #encoder = Encoder(embedding_matrix)
    decoder = Decoder(embedding_matrix)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt)')
    #checkpoint_prefix = '/media/data6t/educationData/submitData/shao/EERNN/model'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, modelr=decoder)

    for epoch in range(0,epochs):
        start = time.time()
        total_loss = 0
        dataset.shuffle(BUFFER_SIZE)
        for batch, (data) in enumerate(dataset):
            loss = 0
            with tf.GradientTape() as tape:
                X,cos_X =  decoder.call_encode(pro_dic)
                #X == (num_problems,bistml_size)
                #data == (batch_size,max_step,2)
                hatt = tf.zeros([BATCH_SIZE,pro_dic.shape[0],LSTM_UNITS])
                data_t = tf.expand_dims(data[:,0,:],1)
                for t in range(1,data.shape[1]):
                    xt = data_forstu(data_t,X)
                    xt = tf.expand_dims(xt,1)
                    # xt == shape(batch_size, 1,4 * LSTM_UNITS)
                    prediction,hatt = decoder.call(xt,data_t,hatt,X,cos_X)
                    data_t = tf.expand_dims(data[:,t,:],1)
                    #data_t == (batch_size,1,2)
                    loss += entroy_loss(prediction, data_t)
            batch_loss = (loss/int(data.shape[1]))
            total_loss += batch_loss
            variables = decoder.bi_lstm.variables + decoder.lstm.variables+decoder.dense2.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            print('.', end='')
            if batch == 0:
                print()
                print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy())
                      )
        end = time.time()
        #checkpoint.save(file_prefix=checkpoint_prefix)
        decoder.save_weights('/media/data6t/educationData/submitData/shao/EERNN/model/my_model_' + epoch + 1)
        print("Epoch {} cost {}".format(epoch + 1, end - start))

        print('start evaluation')
        preds, targets, binary_preds = [], [], []
        X,cos_X =  decoder.call_encode(pro_dic)
        for (batch, data) in enumerate(test_data):
            hatt = tf.zeros([BATCH_SIZE, pro_dic.shape[0], LSTM_UNITS])
            data_t = tf.expand_dims(data[:, 0, :], 1)
            preds, targets, binary_preds = [], [], []
            for t in range(1, data.shape[1]):
                xt = data_forstu(data_t, X)
                xt = tf.expand_dims(xt, 1)
                prediction, hatt = decoder.call(xt, data_t, hatt, X, cos_X)
                data_t = tf.expand_dims(data[:, t, :], 1)
                pred, binary_pred, target_correctness = cal_pre(prediction, data_t)
                preds.append(pred)
                binary_preds.append(binary_pred)
                targets.append(target_correctness)

        preds = np.concatenate(preds)
        binary_preds = np.concatenate(binary_preds)
        targets = np.concatenate(targets)

        np.savetxt("/media/data6t/educationData/submitData/shao/EERNN/result/" + str(epoch + 1) + "preds.txt", preds)
        np.savetxt("/media/data6t/educationData/submitData/shao/EERNN/result/" + str(epoch + 1) + "targets.txt", targets)
        np.savetxt("/media/data6t/educationData/submitData/shao/EERNN/result/" + str(epoch + 1) + "binary_preds.txt", binary_preds)

        #auc_value = roc_auc_score(targets,preds)
        accuracy = accuracy_score(targets, binary_preds)
        precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
        f.write(str(epoch + 1) + ", auc={0}, accuracy={1}, precision={2}, recall={3} \n".format(0, accuracy,
                                                                                                    precision, recall))
        print("\n auc={0}, accuracy={1}, precision={2}, recall={3}".format(0, accuracy, precision, recall))
    f.close()


def evaluate(dataset,embedding_matrix,pro_dic):
    #model.load_weights
    #checkpoint_dir = './training_checkpoints'
    #checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt)')
    #checkpoint = tf.train.load_checkpoint(checkpoint_prefix)
    #model.load_weights(checkpoint_path)
    #status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    optimizer = tf.train.AdamOptimizer(0.001)
    encoder = Encoder(embedding_matrix)
    decoder = Decoder()
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt)')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    preds, targets, binary_preds = list(), list(), list()
    X = encoder(pro_dic)
    cos_X = cosine(X, X)
    for (batch, data) in enumerate(dataset):
        hatt = tf.zeros([BATCH_SIZE, len(pro_dic), LSTM_UNITS])
        data_t = tf.expand_dims(data[:, 0, :], 1)
        for t in range(1, data.shape[1]):
            xt = data_forstu(data_t, X)
            xt = tf.expand_dims(xt, 1)
            prediction, hatt = decoder(xt, data_t, hatt, X, cos_X)
            data_t = tf.expand_dims(data[:, t, :], 1)
            pred, binary_pred,target_correctness= cal_pre(prediction, data_t)
            preds.append(pred)
            binary_preds.append(binary_pred)
            targets.append(target_correctness)

        preds = np.concatenate(preds)
        binary_preds = np.concatenate(binary_preds)
        targets = np.concatenate(targets)
        auc_value = roc_auc_score(targets, preds)
        accuracy = accuracy_score(targets, binary_preds)
        precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
        f.write(str(epoch + 1) + ", auc={0}, accuracy={1}, precision={2}, recall={3} \n".format(auc_value, accuracy,precision, recall))
        print("\n auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy, precision, recall))

if __name__=='__main__':
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.keras.backend.set_session(tf.Session(config=config))
    pro_dic, embedding_matrix,dataset = dataProcess('/media/data6t/educationData/submitData/shao/EERNN/data/HDUout_google.txt', '/media/data6t/educationData/submitData/shao/EERNN/filter_ptiro.txt')
    #train_data,test_data = read_record( "/media/data6t/educationData/submitData/shao/Data/hduuserLC_[15, 1000000, 0.06, 1]_problemLC_[10, 1000000, 0.02, 1]_timeLC_['2016-01-01 23.47.31', '2019-01-02 11.21.49']_OnlyRight_True_EERNN_input.csv")
    train_data, test_data = read_record('/media/data6t/educationData/submitData/shao/EERNNin.txt')
    run(train_data,test_data,pro_dic,embedding_matrix,dataset)
    #evaluate(test_data, embedding_matrix, pro_dic)