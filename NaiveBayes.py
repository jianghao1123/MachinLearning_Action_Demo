# coding=utf-8
from numpy import *


# 训练数据
def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


# 对数据去重
def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set |= set(document)
    return list(vocab_set)


# 创建 目标数据的向量（根据训练数据，如果该词汇出现在训练数据中，训练数据对应的index位标记为1）
def set_of_words_2_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print 'the world %s is not in my vocabulary!' % word
    return return_vec


# 开始训练，根据朴素贝叶斯公式
# p(c|w) = p(w|c)p(c) / p(w)
# 计算各分类下各特征的概率矩阵（因为小数相乘会丢失精度，使用log数据）
def train_nb(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = log(p1_num / p1_denom)
    p0_vect = log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2Clssify, p0Vec, p1Vec, pClass1):
    '''
    分类函数
    :param vec2Clssify: 待分类的矩阵（文档经过set_of_words_2_vec创建的数据向量）
    :param p0Vec:       训练数据在p0分类上各特征的概率
    :param p1Vec:       训练数据在p1分类上各特征的概率
    :param pClass1:     属于该分类的概率
    :return:            分类
    '''
    p1 = sum(vec2Clssify * p1Vec) + log(pClass1)
    p0 = sum(vec2Clssify * p0Vec) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb():
    list_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_posts)
    train_mat = []
    for post_in_doc in list_posts:
        train_mat.append(set_of_words_2_vec(my_vocab_list, post_in_doc))
    p0_v, p1_v, p_ab = train_nb(train_mat, list_classes)

    test_entry = ['so', 'cute', 'garbage', 'stupid']
    doc = set_of_words_2_vec(my_vocab_list, test_entry)
    print classify_nb(doc, p0_v, p1_v, p_ab)


testing_nb()