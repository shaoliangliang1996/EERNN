from hyper_params import *
#import nltk
#from nltk.corpus import stopwords as pw
import re
import os
#coding:utf-8

def preprocess_sentence(w):
    w = re.sub(r"([?.!,])",r" \1 ",w)
    # #数学公式在处理时就没有了 而且也没法预训练  公式处理程序
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z0-9]+", " ", w)
   # w = re.sub(r"[^a-zA-Z+]+"," ",w)
    w = w.rstrip().strip()
    return w
def jieba(infilepath,outfilepath):
    problems = []
    if os.path.exists(outfilepath) == False:
        w = open(outfilepath, 'w')
        stopword = pw.words('english')
        with open(infilepath,'r',encoding='gb18030') as f:
            lines = f.readlines()
            lemmatizer = nltk.WordNetLemmatizer()
            for line in lines:
                line = preprocess_sentence(line)
                pro = ' '.join([lemmatizer.lemmatize(word) for word in str(line).split() if word not in stopword])
            #print(pro)
                problems.append(pro)
                pro = pro+'\n'
                w.writelines(pro)
    else:
        print(outfilepath+'has exist')
        with open(outfilepath, 'r', encoding='gb18030') as f:
            for line in f:
                problems.append(line.strip())
    return problems

def load_glove(word_index):#导入glove的词向量
    embedding_file='data/glove.6B/glove.6B.'
    embedding_file = embedding_file+str(EMBEDDING_DIM)+'d.txt'
    embeddings_index={}#定义字典
    f = open(embedding_file,'r',encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    """转化为矩阵：构建可以加载到embedding层中的嵌入矩阵，形为(max_words（单词数）, embedding_dim（向量维数）) """
    embedding_matrix = np.zeros((NUM_WORDS, EMBEDDING_DIM))
    cantembed = 0
    for word, i in word_index.items():#字典里面的单词和索引
        if i >= NUM_WORDS:continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector#这里只考虑了之前预训练过的词 那些没有的忽略了
        else:
            cantembed +=1
            print('cant  cantembed word',word)

    print('cantembed,embedding_matrix',cantembed,embedding_matrix.shape)
    return embedding_matrix

def dataProcess(infilepath,outfilepath):
    #nltk.download()
    problems = jieba(infilepath, outfilepath)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                   #filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
    tokenizer.fit_on_texts(problems)
    tokenizer.word_index['<pad>'] = 0
    sequences = tokenizer.texts_to_sequences(problems)  # 整数索引的向量化模型
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,padding='post',truncating='post',maxlen=300)
    word_index = tokenizer.word_index  # 索引字典
    print('Found %s unique tokens.' % len(word_index))
    pro_dic = tf.reshape(sequences,[len(sequences),-1])
    print('pro_dic',pro_dic)

    dataset = tf.data.Dataset.from_tensor_slices(sequences)#有不能解决的问题？？？？ 浪费内存
    dataset=dataset.batch(54)
    dataset = dataset.repeat().prefetch(1)

    embedding_matrix = load_glove(word_index)
    return pro_dic,embedding_matrix,dataset






