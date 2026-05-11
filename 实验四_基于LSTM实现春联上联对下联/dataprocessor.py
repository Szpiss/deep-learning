# 加载文本数据
import codecs
import random

import numpy as np
import torch
import json

# 用于生成训练数据
def data_generator(data, batch_size, max_len):
    # 计算每个对联长度的权重
    data_probability = [float(len(x)) for wordcount, [x, y] in data.items()]  # [每个字数key对应对联list中上联数据的个数]
    data_probability = np.array(data_probability) / sum(data_probability)  # 标准化至[0,1]，这是每个字数的权重

    # 随机选择字数，然后随机选择字数对应的上联样本，生成batch
    for idx in range(max_len):
        # 随机选字数id，概率为上面计算的字数权重
        idx = idx + 1
        try:
            size = min(batch_size, len(data[idx][0]))  # batch_size=64，len(data[idx][0])随机选择的字数key对应的上联个数
        except:
            continue
        # 从上联列表下标list中随机选出大小为size的list
        idxs = np.random.choice(len(data[idx][0]), size=size)

        # 返回选出的上联X与下联y, 将原本1-d array维度扩展为(row,col,1)
        yield data[idx][0][idxs], np.expand_dims(data[idx][1][idxs], axis=2)

# 数据读取与切分
def read_data(file_path):
    txt = codecs.open(file_path, encoding='utf-8').readlines()
    txt = [line.strip().split(' ') for line in txt]  # 每行按空格切分
    txt = [line for line in txt if len(line) < 34]  # 过滤掉字数超过maxlen的对联
    max_len = max([len(i) for i in txt])
    return txt, max_len

# 产生数据字典
def generate_count_dict(result_dict, x, y):
    for i, idx in enumerate(x):
        j = len(idx)
        if j not in result_dict:
            result_dict[j] = [[], []]  # [样本数据list,类别标记list]
        result_dict[j][0].append(idx)
        result_dict[j][1].append(y[i])
    return result_dict

# 将字典数据转为numpy
def to_numpy_array(dict):
    for count, [x, y] in dict.items():
        dict[count][0] = np.array(x)
        dict[count][1] = np.array(y)

    return dict

def load_test_data(input_path, output_path, word2idx):
    x, _ = read_data(input_path)
    y, _ = read_data(output_path)
    test_x = []
    test_y = []
    for i, item in enumerate(x):
        word_x = []
        word_y = []
        for word in x[i]:
            try:
                word_x.append(word2idx[word])
            except:
                word_x.append(word2idx['UNK'])
        for word in y[i]:
            try:
                word_y.append(word2idx[word])
            except:
                word_y.append(word2idx['UNK'])
        test_x.append(word_x)
        test_y.append(word_y)
    test_dict = {}

    test_dict = generate_count_dict(test_dict, test_x, test_y)

    test_dict = to_numpy_array(test_dict)
    return test_dict

def load_data(input_path, output_path):

    x, max_len = read_data(input_path)
    y, max_len = read_data(output_path)

    # 获取词表
    vocabulary = x + y
    word_freq_dict = {}
    for words in vocabulary:
        for word in words:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
        word_freq_dict[word] += 1


    word_freq_dict_list = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
    word2idx = {}
    idx2word = {}
    for word, freq in word_freq_dict_list:
        curr_id = len(word2idx)
        word2idx[word] = curr_id
        idx2word[curr_id] = word

    # 训练数据中所有词的个数
    vocab_size = len(word2idx.keys())  # 词汇表大小
    word2idx['UNK'] = vocab_size
    idx2word[vocab_size] = 'UNK'
    with open('./save/word2idx.json', 'w', encoding='utf-8') as f:
        json.dump(word2idx, f)
    with open('./save/idx2word.json', 'w', encoding='utf-8') as f:
        json.dump(idx2word, f)
    # 将x和y转为数值
    train_x = []
    train_y = []
    for sent in x:
        word_arr = []
        for word in sent:
            # liyahu Kiperwasser and Yoav Goldberg. 2016b
            if word_freq_dict[word] > 2:
                pro = 0.8375 / (0.8375 + word_freq_dict[word])
                if random.random() < pro:
                    word = 'UNK'
            word_arr.append(word2idx[word])
        train_x.append(word_arr)
    for sent in y:
        word_arr = []
        for word in sent:
            # liyahu Kiperwasser and Yoav Goldberg. 2016b
            if word_freq_dict[word] > 2:
                pro = 0.8375 / (0.8375 + word_freq_dict[word])
                if random.random() < pro:
                    word = 'UNK'
            word_arr.append(word2idx[word])
        train_y.append(word_arr)



    train_dict = {}

    train_dict = generate_count_dict(train_dict, train_x, train_y)

    train_dict = to_numpy_array(train_dict)

    return train_dict, vocab_size + 1, idx2word, word2idx, max_len


if __name__ == '__main__':
    input_path = './data/test_in.txt'
    output_path = './data/test_out.txt'
    load_data(input_path, output_path)