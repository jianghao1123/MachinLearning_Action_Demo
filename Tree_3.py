# coding=utf-8

import math
import numpy
import operator


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
    :return: 分割后的数组
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
    '''
    海洋生物测试数据集
    特征：浮出水面是否可以生存（1 0）  是否有脚蹼（1 0）
    分类：是否属于鱼类，1是0否，默认数组的最末参数为分类
    :return:
    '''
    data_set = numpy.array([[1, 1, 1]
                     , [1, 1, 1]
                     , [1, 0, 0]
                     , [0, 1, 0]
                     , [0, 1, 0]])
    # 特征
    labels = ['no surfacing', 'flippers']
    return data_set, labels


# 类别占比最多的类
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class = sorted(class_count.iteritems(), key=operator.getitem(1), reverse=True)
    return sorted_class[0][0]


# 创建决策树
def create_tree(data_set, labels):
    '''
    创建决策树
    :param data_set: 数据集
    :param labels:   特征标签
    :return:
    '''

    # 取出所有分类
    class_list = data_set[:, -1]
    # 类别完全相同的情况，直接返回该类别
    if list(class_list).count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完所有特征，返回出现最多的类
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feature = choose_best_feature(data_set)
    best_feature_label = labels[best_feature]
    # 选择最大增益的特征作为父节点
    d_tree = {best_feature_label: {}}
    del(labels[best_feature])
    feature_values = data_set[:, best_feature]
    feature_values = set(feature_values)
    # 遍历该特征的所有取值，递归创建子树
    for value in feature_values:
        sub_labels = labels[:]
        d_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)
    return d_tree

# 待补：决策树的减枝


# test
data, labels = create_data_set()
print data
print calc_shanon_ent(data)
print split_data_set(data, 0, 1)
print choose_best_feature(data)
print create_tree(data, labels)
