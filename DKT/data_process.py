from hyper_params import *


def read_file(dataset_path):
    seqs_by_student = {}  # key为学生编号 值为题目编号，正确与否
    num_skills = 0  # 有多少个问题
    with open(dataset_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            student, problem, is_correct = int(fields[0]), int(fields[1]), int(fields[2])
            num_skills = max(num_skills, problem)
            seqs_by_student[student] = seqs_by_student.get(student, []) + [[problem, is_correct]]  # 这种写法不用if else 判断了，就是扩展每一项字典是这种形式[[1,2],[1,2],[1,2]]
    return seqs_by_student, num_skills + 1

def num_to_one_hot(num, dim):
    base = np.zeros(dim)
    if num >= 0:
        base[num] += 1
    return base

def format_data(seqs, batch_size, num_skills):
    gap = batch_size - len(seqs)
    seqs_in = seqs + [[[0, 0]]] * gap  # pad batch data to fix size
    seq_len = np.array(list(map(lambda seq: len(seq), seqs_in))) - 1
    max_len = max(seq_len)
    x =tf.keras.preprocessing.sequence.pad_sequences([[(j[0] + num_skills * j[1]) for j in i[:-1]] for i in seqs_in], maxlen=max_len,padding='post', value=-1)#输入两层嵌套列表 返回numpy
    input_x = np.array([[num_to_one_hot(j, num_skills*2) for j in i] for i in x],dtype='float32')#????????????????????????????????????????????
    #j[0]为每个学生答题序列去首后 的题目编号  传递的第一个参数形式是[list([1,2,3,4]),list([4,5,6,6])]每个学生的答题编号列表
    target_id = tf.keras.preprocessing.sequence.pad_sequences([[j[0] for j in i[1:]] for i in seqs_in], maxlen=max_len, padding='post', value=0)
    #print('target_id',target_id.shape)
    target_correctness = tf.keras.preprocessing.sequence.pad_sequences([[j[1] for j in i[1:]] for i in seqs_in], maxlen=max_len, padding='post', value=0)
    '''
    print('input__x',type(input_x),input_x.shape)
    print('target_correctness', type(target_correctness), target_correctness.shape)
    print('target_id', type(target_id), target_id.shape)
    print('seq_len, max_len',seq_len, max_len)
    '''
    return input_x, target_id, target_correctness, seq_len, max_len

def traindata_generator(seqs, batch_size, num_skills):
    while 1:
        pos = 0
        end = True
        size = len(seqs)
        while end:
            if pos + batch_size < size:
                batch_seqs = seqs[pos:pos + batch_size]
                pos += batch_size
            else:
                batch_seqs = seqs[pos:]
                pos = size - 1
            if pos >= size - 1:
                end = False
            input_x, target_id, target_correctness, seqs_len, max_len = format_data(batch_seqs, batch_size,
                                                                                    num_skills)
            target_correctness = target_correctness.astype(np.float32)
            #print('input_x，target_id_type,target_correctness', input_x.dtype,target_id.dtype,target_correctness.dtype)
            yield (input_x,(target_id,target_correctness,seqs_len))