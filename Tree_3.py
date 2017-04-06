# coding=utf-8

import math
import numpy


# 计算香农熵
def calc_shanon_ent(data_set):
    length = len(data_set)
    label_count_dic = {}
    shanon_ent = 0.0
    for item in data_set:
        current_label = item[-1]
        if current_label not in label_count_dic.keys():
            label_count_dic[current_label] = 0
        label_count_dic[current_label] += 1
    for label_key in label_count_dic:
        prop = float(label_count_dic[label_key]) / length
        # p*log(p)
        shanon_ent -= prop * math.log(prop, 2)
    return shanon_ent


def split_data_set(data_set, axis, value):
    '''
    截取data_set在给定特征下的数据集合（返回的数据集将剔除该特征）
    :param data_set: 数据集
    :param axis:     该特征所在的列
    :param value:    特征
    :return:
    '''
    ret_data_set = None
    for feature in data_set:
        if feature[axis] == value:
            reduced_feature = list(feature[:axis])
            reduced_feature.extend(list(feature[axis + 1:]))
            if ret_data_set is None:
                ret_data_set = numpy.array(reduced_feature, ndmin=2)
            else:
                ret_data_set = numpy.concatenate((ret_data_set, numpy.array(reduced_feature, ndmin=2)), axis=0)
    return ret_data_set


# 使用信息增益选择最优特征（ 公式：g(D,A) = H(D) - H(D|A)选择最大值）
def choose_best_feature(data_set):
    base_entropy = calc_shanon_ent(data_set)
    # 最大信息增益对应的特征
    best_feature = -1
    # 最大信息增益
    best_info_gain = 0.0
    feature_num = len(data_set[0]) - 1
    for i in range(feature_num):
        # 获得第列的特征数组
        feature_array = data_set[:, i]
        # 过滤重复的特征
        feature_set = set(feature_array)
        new_entropy = 0.0
        for value in feature_set:
            sp_data = split_data_set(data_set, i, value)
            prop = len(sp_data) / float(len(data_set))
            new_entropy += prop * calc_shanon_ent(sp_data)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def create_data_set():
    data_set = numpy.array([[1, 1, 1]
                     , [1, 1, 1]
                     , [1, 0, 0]
                     , [0, 1, 0]
                     , [0, 1, 0]])
    labels = ['no surfacing', 'flippers']
    return data_set, labels


# test
data, labels = create_data_set()
print data
print calc_shanon_ent(data)
print split_data_set(data, 0, 1)
print choose_best_feature(data)
