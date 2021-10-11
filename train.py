import pickle
import keras
from keras.layers import Dropout
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
from tensorflow.keras.models import Model
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
BATCH_SIZE = 80  # 每批大小
NUM_EPOCHS = 20  # 训练周期数
# 创建一个实例
model = Sequential()
# 构建词向量
model.add(Embedding(MAX_FEATURES, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH)) # 输入80 x 80输出80x80x256
model.add(SpatialDropout1D(0.2))
# 构建LSTM层
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
# 来点
model.add(Dropout(0.5))
# 输出层包含四个分类，激活函数设置为'softmax'
model.add(Dense(6, activation="softmax"))
model.add(Activation('softmax'))
# 损失函数设置为分类交叉熵categorical_crossentropy
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练模型
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))

# 评估模型
y_pred = model.predict(Xtest)
y_pred = y_pred.argmax(axis=1)
ytest = ytest.argmax(axis=1)

print('accuracy %s' % accuracy_score(y_pred, ytest))
target_names = ['无情绪', '想买', '伤心', '厌恶', '愤怒', '高兴']
print(classification_report(ytest, y_pred, target_names=target_names))

print("正在保存模型")
model.save('model/my_model.h5')


# 测试模型
print("加载模型")
model = keras.models.load_model('model/my_model.h5')

INPUT_SENTENCES = ['哈哈哈开心', '真是无语，你们怎么搞的', '哥，祝你生日快乐', '你他妈的有病']
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
label2word = {0: '喜悦', 1: '愤怒', 2: '厌恶', 3: '低落'}
for x in model.predict(XX):
    print(x)
    x = x.tolist()
    label = x.index(max(x[0], x[1], x[2], x[3]))
    print(label)
    print('{}'.format(label2word[label]))
