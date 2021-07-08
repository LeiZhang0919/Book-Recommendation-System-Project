# encoding=utf-8


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Input, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import gensim.downloader as api
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from tensorflow.keras import layers
from gensim.models import word2vec
'''

'''


##################################################
# preprocess
##################################################


## The stop_words will be deleted on the TFIDFvectorizer
def text_process(filename):
    regex_web = re.compile(r'http://.*\s+')  # 结尾的.表示任意字符(包括\t\r\f\n) 但是也会匹配空格. 我们这里还好, 都是独立的urls
    pattern = r'[a-zA-Z][-._a-zA-Z]*[a-zA-Z]'  # allow several times of hyphen, dot, single quote such as "Te--rm..ina"

    with open(filename, 'r', encoding='utf-8') as f:
        t1 = f.read().lower()  # 转换为小写
        f.close()
        t2 = re.sub('\n{2,}', '\n\n', t1)  # 将连续2个以上的换行符作为split标准
        print("Line indicator processing: Deleted # characters: ", len(t1) - len(t2))
        t3 = re.sub(regex_web, "", t2)  # 去除网站
        print("Urls processing: Deleted # characters: ", len(t2) - len(t3))
        t4 = t3.split('\n\n')  # 分开段落
        print("# Paragraph: ", len(t4))
        cleaned = [" ".join(re.findall(pattern, para)) for para in
                   t4]  # 找到每个段落中的单词 (可以顺便去除stopword), 还是留在tfidf_vectorizer

        # 去除多余空格
        # 替换数字为x
        return cleaned


## Delete the noise sample, and construct the paragraph; I think the short sentence is not reasonable to regard as a paragraph, so the number of characters threshold is set with 40
def para_filter(dt1, threshold_len=40):
    tt = pd.DataFrame(dt1)
    leng = tt.iloc[:, 0].str.len()
    #     plt.plot(leng)
    deleted_dt = tt[leng < threshold_len]
    print("Under threshold, # possible paragraphs deleted: ", len(deleted_dt))
    samples = tt[leng >= threshold_len].reset_index(drop=True)
    return samples, deleted_dt

# dt1 = text_process(wd+filename[0])
# dtt1,deleted_dt1 = para_filter(dt1, threshold_len=100)
# deleted_dt1

# # DataSet construct
# dttt = pd.DataFrame(np.empty((0,1),float)) # Final dataset

# for i in range(len(filename)):
#     print("\n\n========================================================")
#     dt1 = text_process(wd+filename[i])
#     dtt1,deleted_dt1 = para_filter(dt1, threshold_len=5) # 避免稀疏!!!!

#     print('\nLabel:', str(i),'\nFor this document', filename[i],  "it perserved # of samples: ", dtt1.shape[0],'\n')
#     dtt1['label'] = i
#     dttt = pd.concat([dttt,dtt1], axis=0, ignore_index=True)


# dttt







##################################################
# EDA
##################################################

def eda_MAX_NB_WORDS(corpus, ratio=0.95, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', char_level=False):
    '''
    Input: list of sentences (string type)
    '''
    ### 确认词长
    # ratio = 0.95
    # corpus = train_5k


    tokenizer_eda = Tokenizer(num_words=None, filters=filters, lower=True, char_level=char_level) # 如果有这个, NLTK的preprocessing可以不用做
    tokenizer_eda.fit_on_texts(corpus)
    b = pd.DataFrame(tokenizer_eda.word_counts.items(), columns=['word','count'])
    a = b.sort_values(by='count', ascending=False).reset_index() # 排序重建index 就是 tokenizer中的word_index


    ############# 累加百分比 可视化
    plt.figure(figsize=(20,5))
    word_distribution = a['count'].cumsum() / a['count'].sum() # 求累加百分比
    word_distribution.plot() # 出图

    # cut_index = np.argmin(abs(word_distribution-ratio)) # 找到离0.8最近的index位置
    diff = abs(word_distribution - ratio)
    cut_index = diff[diff==min(diff)].index[0]

    plt.plot([cut_index,cut_index], [0,ratio]) # 找出 固定 ratio 的index
    plt.plot([0, cut_index], [ratio,ratio])
    plt.xlabel("word_index") # 需要先sort, 才能说是index of words.
    plt.ylabel("word_cum_counts_perc")
    plt.title("MAX_NB_WORDS Cumsum Percentage")
    plt.show()


    #############  大概取词范围 可视化
    plt.figure(figsize=(20,5))
    b.iloc[:,1].plot() # 出图
    plt.plot([cut_index, cut_index], [0, max(b['count'])]) # 找出 固定 ratio 的index
    plt.plot([0, cut_index], [max(b['count']), max(b['count'])])
    plt.xlabel("word_index") # 需要先sort, 才能说是index of words.
    plt.ylabel("word_count")
    plt.title("MAX_NB_WORDS Percentage")
    plt.show()
    print("Cut index with", ratio*100, "% of corpus: ", cut_index, '\n')
    # stopwords?
    print(a.sort_values(by='count', ascending=False).head(20))

    return int(cut_index)
    # return int(cut_index)+1

# eda_MAX_NB_WORDS(corpus = filtered_corpus, ratio = 0.95)



def eda_MAX_DOC_LEN(corpus, ratio=0.9, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', char_level=False):
    '''
    Input: list of sentences (string type)
    '''

    # MAX_DOC_LEN=30
    # corpus = train_5k
    tokenizer_eda = Tokenizer(num_words=None, filters=filters,
                              lower=True, char_level=char_level)
    tokenizer_eda.fit_on_texts(corpus)

    dt_q1 = pd.DataFrame([len(i) for i in tokenizer_eda.texts_to_sequences(corpus)], columns=['length'])

    c = dt_q1['length'].value_counts().sort_index()  # 频数统计, 且按index重新排序
    sent_cdf = c.cumsum() / c.sum()
    sent_pdf = c / c.sum()
    # cut_index = np.argmin(abs(sent_cdf - ratio))  # 找到离0.8最近的index位置
    diff = abs(sent_cdf - ratio)
    cut_index = diff[diff==min(diff)].index[0]

    ############# 累加百分比 可视化
    plt.figure(figsize=(20, 5))
    sent_cdf.plot()  # 出图

    plt.plot([cut_index, cut_index], [0, ratio])  # 找出 固定 ratio 的index
    plt.plot([0, cut_index], [ratio, ratio])
    plt.xlabel("word_length")  # 需要先sort, 才能说是index of words.
    plt.ylabel("word_cum_counts_perc")
    plt.title("MAX_DOC_LEN CDF")
    plt.show()

    plt.figure(figsize=(20, 5))
    sent_pdf.plot()  # 出图
    plt.plot([cut_index, cut_index], [0, max(sent_pdf)])  # 找出 固定 ratio 的index
    plt.plot([0, cut_index], [max(sent_pdf), max(sent_pdf)])  # 横线
    plt.xlabel("word_length")  # 需要先sort, 才能说是index of words.
    plt.ylabel("word_counts_perc")
    plt.title("MAX_DOC_LEN PDF")
    plt.show()

    print("Cut index with", ratio * 100, "% of corpus: ", cut_index)
    return int(cut_index)

# eda_MAX_DOC_LEN(corpus = filtered_corpus, ratio=0.9)










##################################################
# Dataset prepare class
##################################################

class text_preprocessor(object): # stopwords 在NLTK的 tokenizer中才有. 所以用keras的话, 就预处理的时候, 把stopwords去掉

    def __init__(self, doc_len, max_words, docs, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',char_level=False, zero_pad = '_'):
        '''
        initialize a processor

        input: a sequence of string (Training dataset)
        processor = text_preprocessor(MAX_DOC_LEN, MAX_NB_WORDS, sentences_train)

        '''

        self.MAX_DOC_LEN = doc_len
        self.MAX_NB_WORDS = max_words
        self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, filters=filters, char_level=char_level)
        self.tokenizer.fit_on_texts(docs)
        self.corpus = docs

        if zero_pad: # for cnn this is not needed
            self.tokenizer.word_index[zero_pad] = 0
            self.tokenizer.index_word[0] = zero_pad


        self.index_word = self.tokenizer.index_word
        self.word_index = self.tokenizer.word_index



    def __repr__(self): # print(自己的时候)出来下面的东西
        return 'A class which has method:\n' \
               'generate_seq(sentences_train) \n' \
               'w2v_pretrain(dimension of embedding)\n' \
               'load_glove_w2v(dimension of embedding)\n' \
               ''

    def generate_seq(self, docs, padding='post', truncating='post'):
        sequences = self.tokenizer.texts_to_sequences(docs)
        # padded_sequences = pad_sequences(sequences, maxlen=self.MAX_DOC_LEN, padding='post')
        padded_sequences = pad_sequences(sequences, maxlen=self.MAX_DOC_LEN, padding=padding, truncating=truncating)
        # 这里不用+1
        return padded_sequences

    def w2v_pretrain(self, EMBEDDING_DIM, min_count=1, seed=1, cbow_mean=1,
                                 negative=5, window=5, iter=30,
                                 workers=8): #

        ## Generate pretrained Embedding with all of tokens in training sentences
        wv_model = word2vec.Word2Vec(sentences=self.corpus, min_count=min_count, seed=seed, cbow_mean=cbow_mean,
                                     size=EMBEDDING_DIM, negative=negative, window=window, iter=iter,
                                     workers=workers)  # Based on tokens in all sentences, training the W2V # sg = 1 为 skipgram

        NUM_WORDS = min(self.MAX_NB_WORDS,
                        len(self.tokenizer.word_index))  # Keep the # of highest freq of words, in this case : 2500
        embedding_matrix = np.zeros((NUM_WORDS + 1, EMBEDDING_DIM))  # "+1" is for padding symbol that equal 0
#         embedding_matrix = np.zeros((NUM_WORDS, EMBEDDING_DIM))  #  RNN not needed

        for word, i in self.tokenizer.word_index.items():
            if i >= NUM_WORDS:
                continue
            if word in wv_model.wv:
                embedding_matrix[i] = wv_model.wv[word]  # load pretrained embedding on my indexing table embedding

        PRETRAINED_WORD_VECTOR = embedding_matrix

        return PRETRAINED_WORD_VECTOR

    def load_glove_w2v(self, EMBEDDING_DIM):
        word_vectors = api.load(
            "glove-wiki-gigaword-" + str(EMBEDDING_DIM))  # load pre-trained word-vectors from gensim-data

        NUM_WORDS = min(self.MAX_NB_WORDS, len(self.tokenizer.word_index))  # Keep the # of highest freq of words, in this case : 2500

        embedding_matrix = np.zeros((NUM_WORDS + 1, EMBEDDING_DIM)) # "+1" is for padding symbol that equal 0
#         embedding_matrix = np.zeros((NUM_WORDS, EMBEDDING_DIM))  #  RNN not needed

        for word, i in self.tokenizer.word_index.items():
            if i >= NUM_WORDS:  # index 超过2500 扔掉
                continue
            if word in word_vectors.wv:  # "i" in word_vectors.wv
                embedding_matrix[i] = word_vectors[word]  # load pretrained embedding on my indexing table embedding

        PRETRAINED_WORD_VECTOR = embedding_matrix

        return PRETRAINED_WORD_VECTOR



##################################################
# define CNN part
##################################################



def cnn_model(FILTER_SIZES, MAX_NB_WORDS, MAX_DOC_LEN, NAME='cnn_base', EMBEDDING_DIM=200, NUM_FILTERS=64,
              PRETRAINED_WORD_VECTOR=None, trainable_switch=True, bert_embedding=True):
    model = None

    main_input = Input(shape=(MAX_DOC_LEN,), dtype='int32', name='main_input')

    if (PRETRAINED_WORD_VECTOR is not None):
        embed_1 = Embedding(input_dim=MAX_NB_WORDS , output_dim=EMBEDDING_DIM, embeddings_initializer='uniform',
                            input_length=MAX_DOC_LEN, name='pretrained_embedding_trainable'
                            , weights=[PRETRAINED_WORD_VECTOR], trainable=trainable_switch)(main_input)

    else: # 默认trainable
        embed_1 = Embedding(input_dim=MAX_NB_WORDS , output_dim=EMBEDDING_DIM, embeddings_initializer='uniform',
                            input_length=MAX_DOC_LEN, name='embedding_trainable'
                            , trainable=True)(main_input)

        # 这个+1 留到外面做
        # embed_1 = Embedding(input_dim=MAX_NB_WORDS + 1, output_dim=EMBEDDING_DIM, embeddings_initializer='uniform',
        #                     input_length=MAX_DOC_LEN, name='embedding_trainable'
        #                     , trainable=True)(main_input)    # Convolution-pooling-flat block
    conv_blocks = []
    for f in FILTER_SIZES:  # For every filter
        conv = Conv1D(filters=NUM_FILTERS, kernel_size=f, name='conv_' + str(f) + '_gram', strides=1,
                      activation='relu')(
            embed_1)  # convolution  # filter-kernal extracting 64 features with ReLU activation function
        pool = MaxPooling1D(pool_size=MAX_DOC_LEN - f + 1, name='pool_' + str(f) + '_gram')(
            conv)  # maxpooling size = MAX_DOC_LEN - filter_size + 1
        flat = Flatten(name='flat_' + str(f) + '_gram')(
            pool)  # flatten filters extracting features (size*number = 3*64)
        conv_blocks.append(flat)

    if len(conv_blocks) > 1:
        z = Concatenate(name='concate')(conv_blocks)  # Concatenate的 input 是一个 list [flat_1, flat_2, flat_3]
    else:
        z = conv_blocks[0]

    #     pred = Dense(3, activation='softmax')(z)
    model = Model(inputs=main_input, outputs=z, name=NAME)

    return model

# testing
# cnn_base = cnn_model(FILTER_SIZES=[2,3,4], NUM_FILTERS=64, MAX_DOC_LEN=MAX_DOC_LEN, MAX_NB_WORDS=MAX_NB_WORDS, EMBEDDING_DIM=300)
# cnn_base.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# cnn_base.fit(x=X_test,y=y_test,batch_size=20)
# cnn_base.summary()
# plot_model(cnn_base,show_shapes=True)




##################################################
# history plot
##################################################


def history_plot(training, extra_metric=None):

    ################## plot training history
#     dic = ['val_loss', 'loss', 'val_acc', 'acc', "val_auroc"] # print(training.history)
#     loss: 0.8109 - acc: 0.6362 - auroc: 0.7960 - val_loss: 0.6793 - val_acc: 0.7144 - val_auroc: 0.8684
    dic = list(training.history.keys())

    if extra_metric is not None:
        idx = [[0,3],[1,4],[2,5]]
    else:
        idx = [[0,2],[1,3]]

    for i,j in idx:
        print("========================================================================")
        print(dic[i],dic[j])
        xx = list(range(1,len(training.history[dic[i]])+1))
        plt.plot(xx,training.history[dic[i]], color = 'navy', lw = 2, label = 'Model_'+str(dic[i]))
        plt.plot(xx,training.history[dic[j]], color = 'darkorange', lw = 2, label = 'Model_'+str(dic[j]))
        plt.title(str(dic[i]) + "v.s. training_" + str(dic[j]))
        plt.xlabel('Epochs')
        plt.ylabel(str(dic[i]))
        plt.legend()
        plt.show();
    return None
