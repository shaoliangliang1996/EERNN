from hyper_params import *
from TensorFlowDKT import *
from data_process import *
import time
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score


def crossentropy_loss(target_id, target_correctness, logits,num_skills=124):
    flat_logits = tf.reshape(logits, [-1]) 
    flat_target_correctness = tf.reshape(target_correctness, [-1]) 

    flat_base_target_index = tf.range(BATCH_SIZE * target_id.shape[-1]) * num_skills
    flat_bias_target_id = tf.reshape(target_id, [-1])
    flat_target_id = flat_bias_target_id + flat_base_target_index  
    flat_target_logits = tf.gather(flat_logits,flat_target_id)
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness,logits=flat_target_logits))

def cal_pre(target_id, logits,num_skills=124):
    flat_logits = tf.reshape(logits, [-1]) 
    max_steps = target_id.shape[-1]
    flat_base_target_index = tf.range(BATCH_SIZE * max_steps) * num_skills
    flat_bias_target_id = tf.reshape(target_id, [-1])
    flat_target_id = flat_bias_target_id + flat_base_target_index 
    flat_target_logits = tf.gather(flat_logits, flat_target_id)
    pred = tf.sigmoid(tf.reshape(flat_target_logits, [BATCH_SIZE, max_steps]))
    binary_pred = tf.cast(tf.greater_equal(pred, 0.5), tf.int32)  

    return pred,binary_pred
def run(train_seqs,num_skills):
    # process data
    np.random.shuffle(train_seqs)
    #print('train_num_student',type(train_seqs),len(train_seqs))
    step = int(len(train_seqs)/BATCH_SIZE)+1
    config = {"hidden_neurons": LSTM_UNITS ,
              "batch_size": BATCH_SIZE,
              "keep_prob": 0.6,
              'dense_unit': num_skills}
    model = DKTModel(config)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
    EPOCHS =20
    batch_data = traindata_generator(train_seqs, BATCH_SIZE, num_skills)
    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        for i in range(step):
            input_data = next(batch_data)
            input_x, target_id, target_correctness = input_data[0],input_data[1][0],input_data[1][1]
            with tf.GradientTape() as tape:
                    x = model(input_x)
                    loss = crossentropy_loss(target_id, target_correctness, x,124)
            total_loss += loss
            variables = model.lstm.variables+model.dense.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables),tf.train.get_or_create_global_step())
            print(".", end='', flush=True)
            if i % 10 == 0:
                print()
                print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1,
                                                             i,
                                                             loss)
                      )
        end = time.time()
        print("Epoch {} cost {}".format(epoch + 1, end - start))

    print("End training...")
    # Save weights to a TensorFlow Checkpoint file
    model.save_weights('./weights/my_model')

def test(test_seqs,num_skills):
    # Restore the model's state,
    # this requires a model with the same architecture.
    config = {"hidden_neurons": LSTM_UNITS,
              "batch_size": BATCH_SIZE,
              "keep_prob": 0.6,
              'dense_unit': num_skills}
    model = DKTModel(config)
    preds, targets ,binary_preds= list(), list(),list()
    batch_data = traindata_generator(test_seqs, BATCH_SIZE, num_skills)
    model.load_weights('./weights/my_model')
    num = int(len(test_seqs)/BATCH_SIZE)+1

    for i in range(num):
        input_data = next(batch_data)
        input_x, target_id, target_correctness ,seqs_len= input_data[0], input_data[1][0], input_data[1][1],input_data[1][2]
        x = model(input_x)
        pred, binary_pred = cal_pre(target_id, x, num_skills=124)
        for j in range(len(seqs_len)):
            preds.append(pred[j, 0:seqs_len[j]])
            binary_preds.append(binary_pred[j, 0:seqs_len[j]])
            targets.append(target_correctness[j, 0:seqs_len[j]])

    preds = np.concatenate(preds)  # 对于列表 是列表的链接
    binary_preds = np.concatenate(binary_preds)
    targets = np.concatenate(targets)  # 数组拼接 默认axis=0 进行横向拼接，就是扩展行数
    auc_value = roc_auc_score(targets, preds)  # 现成的函数来测评模型
    accuracy = accuracy_score(targets, binary_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
    print("\n auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy, precision, recall))
    
if __name__ == "__main__":
    seqs_by_student, num_skills = read_file('./data/assistments_fu.txt')
    # print('num_student',len(seqs_by_student),num_skills)
    train_seqs, test_seqs = train_test_split(seqs_by_student, test_size=0.2, random_state=0)
    run(train_seqs,num_skills)
    test(test_seqs,num_skills)
