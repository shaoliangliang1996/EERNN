import tensorflow as tf
import os
import re
import pickle
import numpy as np
import nltk
#from nltk.corpus import stopwords as pw

class OJProblemtextProcessor(object): #数据集名称
    DataName = ""
    #临时文件夹位置
    TmpDir = ""
    word2Id = {}
    problemName2problemId = {}
    problemId2problemName = {}
    problemList = []
    def __init__(self,userLC,problemLC,timeLC,OnlyRight,DataName = 'hdu',TmpDir = '/home/shaoliangliang/code/MANN_TCA_SA/datapart/'):
        self.DataName = DataName
        self.TmpDir = TmpDir
        self.LC2Str = ('userLC_' + str(userLC) + '_problemLC_' + str(problemLC) + '_timeLC_' + re.sub(r':', '.', str(timeLC)) + '_OnlyRight_' + str(OnlyRight))
        self.InputprobemPath = self.TmpDir + self.DataName + '_problem.txt'
        self.filerprobemPath = self.TmpDir + self.DataName +self.LC2Str+ '_filtered_'+'problem.txt'
        self.embedding_file = self.TmpDir + 'glove.6B.'
        self.word2IdPath = self.TmpDir + self.DataName+'_word2Id_' +self.LC2Str+'.txt'
        self.cantembedding = self.TmpDir +'cantembedding.txt'
        self.RawKnowledge2Problem = self.TmpDir + self.DataName+'_RawKnowledge2Problem.txt'

    def InitialData(self):
        f1 = open(self.TmpDir + self.DataName + self.LC2Str+ '_problemName2problemId.pkl', 'rb')
        f2 = open(self.TmpDir + self.DataName + self.LC2Str + '_problemId2problemName.pkl', 'rb')
        self.problemName2problemId = pickle.load(f1)
        self.problemId2problemName = pickle.load(f2)
        f1.close(), f2.close()
        print('self.problemName2problemId',len(self.problemName2problemId))
    def Sentence2Words(self,raw_str):
        lemmatizer = nltk.WordNetLemmatizer()  # lemmatizer.lemmatize(word)
        # re_sentence
        raw_str = re.sub(r'([+])',' add ',raw_str)
        raw_str = re.sub(r'([-])',' sub ',raw_str)
        raw_str = re.sub(r"([?.!,])", r" \1 ", raw_str)
        # #数学公式在处理时就没有了 而且也没法预训练  公式处理程序
        raw_str = re.sub(r'[" "]+', " ", raw_str)
        raw_str = re.sub(r"[^a-zA-Z0-9]+", " ", raw_str)
        raw_str = raw_str.strip()
        # split2word
        sentence = ' '.join([word.lower() for word in str(raw_str).split() if word.lower() not in self.stopword])
        return sentence

    def Write2File(self):
        with open( self.word2IdPath,'w') as f:
            for k,v in  self.word2Id.items():
                f.writelines(str(k)+' '+str(v)+'\n')

    def SortandInitialProblembyId(self):
        self.InitialData()
        if os.path.exists(self.filerprobemPath):
            print(self.filerprobemPath + 'has exist')
            with open(self.filerprobemPath, 'r') as f:
                for line in f:
                    self.problemList.append(line.strip())
        else:
            w_f = open(self.filerprobemPath, 'w')
            stopword = pw.words('english')
            with open(self.cantembedding, 'r') as f:
                for word in f:
                    stopword.append(word.strip())
            print('stopword', len(stopword))
            problemdic = {}
            index = 1000
            with open(self.InputprobemPath,'r',encoding='gb18030') as f:
                for line in f:
                    if index in self.problemName2problemId.keys():
                        problemdic[self.problemName2problemId[index]]=self.Sentence2Words(line)
                    index +=1
            problemdic = sorted(problemdic.items(),key=lambda x:x[0])
            w_f.writelines('\n')
            self.problemList.append([])
            for t in problemdic:
                k,v = t
                if len(v)==0:
                    print('has empty problem',self.problemId2problemName[k])
                else:
                    self.problemList.append(v)
                    w_f.writelines(v+'\n')
            w_f.close()

    def LoadGlove(self,EMBEDDING_DIM):  # 导入glove的词向量
        embedding_file = self.embedding_file + str(EMBEDDING_DIM) + 'd.txt'
        embeddings_index = {}  # 定义字典
        f = open(embedding_file, 'r', encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        """转化为矩阵：构建可以加载到embedding层中的嵌入矩阵，形为(max_words（单词数）, embedding_dim（向量维数）) """
        embedding_matrix = np.zeros((len(self.word2Id), EMBEDDING_DIM))
        cantembed = 0
        for word, i in self.word2Id.items():  # 字典里面的单词和索引
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                cantembed += 1
                print(word)

        # print('cantembed,embedding_matrix',cantembed)
        return embedding_matrix

    def Problem2Tensor(self,EMBEDDING_DIM=50,Maxlen=100):
        #produces pro_dic,embedding_matrix without labels used for EERNN
        self.SortandInitialProblembyId()
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=4000,split=" ",char_level=False)
        tokenizer.fit_on_texts(self.problemList)
        tokenizer.word_index['<pad>'] = 0
        sequences = tokenizer.texts_to_sequences(self.problemList)  # 整数索引的向量化模型
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', truncating='post',maxlen=Maxlen)
        self.word2Id = tokenizer.word_index  # 索引字典
        print('Found %s unique tokens.' % len( self.word2Id))
        pro_dic = tf.reshape(sequences, [len(sequences), -1])
        embedding_matrix =  self.LoadGlove(EMBEDDING_DIM)
        self.Write2File()
        return pro_dic,embedding_matrix

    def ProblemwithLabel2Dataset(self,BATCH_SIZE=482,PREFETCH_SIZE=128,CPU_CORES=4,EMBEDDING_DIM=100,Maxlen=256):
        #produces pro_dic,embedding_matrix with labels used for MANN
        pro_dic, embedding_matrix = self. Problem2Tensor(EMBEDDING_DIM,Maxlen)
        def merge_function(*args):
            return args
        problemId2label = {}
        with open(self.RawKnowledge2Problem,'r') as f:
            num = 0
            for line in f:
                tmp = line.strip().split(',')
                label = tmp[0]
                pro = tmp[1:]
                for p in pro:
                   if int(p) in self.problemName2problemId.keys():
                        problemId2label[self.problemName2problemId[int(p)]]=int(label)
        print('dic',len(problemId2label))
        num_problem = len(problemId2label)

        pro_list = []
        pro_label_list = []
        sim_pro_list = []
        sim_pro_label = []
        sort_label = []
        tmp = sorted(problemId2label.items(), key=lambda x: x[0])
        for t in tmp:
            k,v =t
            sort_label.append(v)
        print('sort_label',len(sort_label))
        for i in range(0,num_problem):
            pro_list +=[i]*num_problem
            pro_label_list += [problemId2label[i]]*num_problem
        print('pro_list,pro_label_list ',len(pro_list),len(pro_label_list))
        for i in range(0,num_problem):
            sim_pro_list += range(0,num_problem)
            sim_pro_label += sort_label
        print('sim_pro,sim_pro_label',len(sim_pro_list),len(sim_pro_label))

        pro_list = tf.reshape(pro_list, [1, num_problem * num_problem])
        pro_label_list = tf.reshape(pro_label_list, [1, num_problem * num_problem])
        sim_pro = tf.reshape(sim_pro_list, [1, num_problem * num_problem])
        sim_pro_label = tf.reshape(sim_pro_label, [1, num_problem * num_problem])
        data_tokens = tf.gather(pro_dic, pro_list, axis=0)
        data_sim_tokens = tf.gather(pro_dic, sim_pro, axis=0)
        print('data_tokens,data_sim_tokens', data_tokens.shape, data_sim_tokens.shape)
        dataset_tokens = tf.data.Dataset.from_tensor_slices(data_tokens[0, :, :])
        dataset_sim_tokens = tf.data.Dataset.from_tensor_slices(data_sim_tokens[0, :, :])
        dataset_tags = tf.data.Dataset.from_tensor_slices(pro_label_list[0, :])
        dataset_sim_tags = tf.data.Dataset.from_tensor_slices(sim_pro_label[0, :])
        print('dataset_tokens', dataset_tokens, dataset_sim_tags)
        dataset_tokens = dataset_tokens.padded_batch(BATCH_SIZE, padded_shapes=[256])
        dataset_sim_tokens = dataset_sim_tokens.padded_batch(BATCH_SIZE, padded_shapes=[256])
        dataset_tags = dataset_tags.padded_batch(BATCH_SIZE, padded_shapes=[])
        dataset_sim_tags = dataset_sim_tags.padded_batch(BATCH_SIZE, padded_shapes=[])

        dataset_zipped_sim = tf.data.Dataset.zip((
            dataset_tokens, dataset_tags,
            dataset_sim_tokens, dataset_sim_tags
        ))
        dataset_merged_sim = dataset_zipped_sim.map(merge_function, num_parallel_calls=CPU_CORES)
        dataset_sim = dataset_merged_sim.prefetch(PREFETCH_SIZE)
        return dataset_sim,embedding_matrix

if __name__ == "__main__":
   #nltk.download('stopwords')
   k = OJProblemtextProcessor([15,1000000,0.06,1],[10,1000000,0.02,1],['2005-01-01 23:47:31','2019-01-02 11:21:49'],OnlyRight=True)
   pro_dic, embedding_matrix = k.Problem2Tensor()
   print('pro_dic, embedding_matrix', pro_dic.shape,embedding_matrix.shape)
   #dataset_sim, embedding_matrix = k.ProblemwithLabel2Dataset(BATCH_SIZE=482,PREFETCH_SIZE=128,CPU_CORES=4,EMBEDDING_DIM=50)
   #print('dataset_sim',dataset_sim,len(embedding_matrix))
