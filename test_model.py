import pickle
import keras
from keras.layers.core import Activation, Dense, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import jieba  # 用来分词
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# 加载分词字典
print("正在加载分词字典")
with open('model/word_dict.pickle', 'rb') as handle:
    word2index = pickle.load(handle)

# 准备数据
print("正在准备数据")
MAX_FEATURES = 40002  # 最大词频数
MAX_SENTENCE_LENGTH = 80  # 句子最大长度
num_recs = 0  # 样本数

with open("data/train.json", "r", encoding="utf-8",errors='ignore') as f:
    lines = json.load(f)
    f.close()
    for line in lines:
        num_recs += 1

# 初始化句子数组和label数组
X = np.empty(num_recs, dtype=list)
y = np.zeros(num_recs)
i = 0

with open("data/train.json", "r", encoding="utf-8", errors='ignore') as f:
    lines = json.load(f)
    f.close()
    for line in lines:
        sentence = line[0].replace(' ', '')
        label = line[1]
        words = jieba.cut(sentence)
        seqs = []
        for word in words:
            # 在词频中
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"]) # 不在词频内的补为UNK
        X[i] = seqs
        y[i] = int(label)
        i += 1

# 把句子转换成数字序列，并对句子进行统一长度，长的截断，短的补0
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
# 使用pandas对label进行one-hot编码
y1 = pd.get_dummies(y).values
print(X.shape)
print(y1.shape)
# 数据划分 8020法则
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y1, test_size=0.2, random_state=42)
# 网络构建
EMBEDDING_SIZE = 256  # 词向量维度
HIDDEN_LAYER_SIZE = 128  # 隐藏层大小
BATCH_SIZE = 64  # 每批大小
NUM_EPOCHS = 5  # 训练周期数

# 测试模型
print("加载模型")
model = keras.models.load_model('model/my_model.h5')

INPUT_SENTENCES = ['哈哈哈开心', '真是无语你们怎么搞的', '这只基金非常不错值得买我要加仓', '医疗行业最近不行了，千万不要再加仓了什么都别买了！', '稳赔不赚，哎呦我可赔死了，绝了', '你是个笨蛋么你太让我伤心了']
XX = np.empty(len(INPUT_SENTENCES), dtype=list)
i = 0
for sentence in INPUT_SENTENCES:
    words = jieba.cut(sentence)
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i += 1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
label2word = {0: '无感', 1: '想买', 2: '伤心', 3: '厌恶', 4: '愤怒', 5: '高兴'}
for x in model.predict(XX):
    print(x)
    x = x.tolist()
    label = x.index(max(x[0], x[1], x[2], x[3], x[4], x[5]))
    print(label)
    print('{}'.format(label2word[label]))
