from hyper_params import *
from sklearn.model_selection import train_test_split
def read_record(filename):
    pro_dic = []
    stu = -1
    with open(filename,'r')as f:
        lines = f.readlines()
        for line in lines:
            stu_id,pro_id,cor = int(line.split(" ")[0]),int(line.split(" ")[1]),int(line.split(" ")[2])
            if stu_id != stu:#这里还可以加速？？
                pro_dic.append([])
                stu +=1
            pro_dic[stu] += [[pro_id,cor]]#

    print('how many stu',len(pro_dic))
    tmp = tf.keras.preprocessing.sequence.pad_sequences(pro_dic,value=-1,padding='post',truncating='post',maxlen=MAXLEN)
    train_data, test_data = train_test_split(tmp, test_size=0.2, random_state=0)
    dataset_train = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size=tf.constant(SHUFFLE_BUFFER_SIZE, dtype=tf.int64))
    dataset_test = tf.data.Dataset.from_tensor_slices(test_data)
    dataset_train = dataset_train.repeat().prefetch(PREFETCH_SIZE).batch(BATCH_SIZE,drop_remainder=True)
    dataset_test = dataset_test.prefetch(PREFETCH_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    return dataset_train,dataset_test



