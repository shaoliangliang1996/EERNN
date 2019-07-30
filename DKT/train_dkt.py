from hyper_params import *
import tensorflow as tf
#tf.enable_eager_execution()
from TensorFlowDKT import *
from data_process import *
NUM_skill =0

def crossentropy_loss(true, logits,num_skills=124):
    print("EREWRFDSSSSSSSSSSSSSSSSSSSSSSS",true[0],true[1])
    target_id, target_correctness = true[0],true[1]
    print('target_id, target_correctness,logits',target_id.shape, target_correctness.shape,logits.shape,target_id.dtype)#两种方法维度不统一
    flat_logits = tf.reshape(logits, [-1])  # 展平 变一维
    flat_target_correctness = tf.reshape(target_correctness, [-1])  # 展平

    print('flat_target_correctness',NUM_skill,flat_target_correctness.shape)
    #flat_base_target_index = tf.range(BATCH_SIZE) * num_skills  # 创建一个数字序列,该数字开始于 start  之所以这么写是因为那个logits的维度这样的
    li = []
    for tm in range(BATCH_SIZE):
        li.append([tm] * target_id.shape[1])
    flat_base_target_index = np.asarray(li) * num_skills  # 更改 理解有误？？？？
    a = tf.convert_to_tensor(flat_base_target_index)
    flat_base_target_index = tf.reshape(a, [-1])
    flat_bias_target_id = tf.reshape(target_id, [-1])
    print('flat_bias_target_id + flat_base_target_index ',flat_bias_target_id.dtype,flat_base_target_index.dtype )
    flat_target_id = flat_bias_target_id + flat_base_target_index  # 和数据处理时的操作对应
    print('flat_target_id_type', flat_target_id.dtype)
    flat_target_logits = tf.gather(flat_logits,flat_target_id)  # 取切片 去特定用户的预测  一个没有sigmod的值拿来做loss 这个值长啥样？？？？？？？？这样更能体现细节？？？ 其实拿sigmod完的数据来做也行？？？？？？？
    print('11111')
    #pred = tf.sigmoid(tf.reshape(flat_target_logits, [BATCH_SIZE, max_steps]))
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness,logits=flat_target_logits))

    #return 0
def run():
    # process data
    seqs_by_student, num_skills = read_file('assistments.txt')
    NUM_skill = num_skills
    print('num_student',len(seqs_by_student),num_skills)
    train_seqs, test_seqs=train_test_split(seqs_by_student, test_size=0.2, random_state=0)
    #train_seqs, test_seqs = split_dataset(seqs_by_student)
    print('train_num_student',type(train_seqs),len(train_seqs),len(test_seqs))
    step = int(len(train_seqs)/BATCH_SIZE)+1
    config = {"hidden_neurons": LSTM_UNITS ,
              "batch_size": BATCH_SIZE,
              "keep_prob": 0.6,
              "input_size": num_skills * 2,
              'num_skills':num_skills,
              'dense_unit': num_skills}
    model = DKTModel(config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, clipnorm=CLIP_NORM),
        loss=crossentropy_loss,
        metrics=[]
    )
    model.fit_generator(traindata_generator(train_seqs,BATCH_SIZE, num_skills),epochs=1,verbose=1,steps_per_epoch=step)
#还少了模型保存
















if __name__ == "__main__":
    #arg_parser = argparse.ArgumentParser(description="train dkt model")
    #arg_parser.add_argument("--dataset", dest="dataset", required=True)
    #args = arg_parser.parse_args()
    run()
