# coding=utf-8
# 最简单的K-近邻算法


from numpy import *
import operator


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, data_set, labels, k):
    '''
    K近邻算法
    :param inx: 传入数据
    :param data_set: 原数据集
    :param labels: 数据分类集
    :param k: K参数
    :return: 返回分类
    '''

    # 获取数据集的行数
    data_set_size = data_set.shape[0]
    # 将inx data_set_size行扩展，并data_set相减
    # 下面要用欧式距离公式计算inx与数据集中每条数据的距离
    diff_mat = tile(inx, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    # 将矩阵的每一行向量相加
    sq_distance = sq_diff_mat.sum(axis=1)
    distance = sq_distance ** 0.5
    # 将 distance从小到大排列，返回索引集合
    sort_dist_indicies = distance.argsort()
    class_count = {}
    for i in range(k):
        # 从左往右（即从小到大）遍历K个元素，取得该元素下的特征分类label，存入class_count字典中
        vote_label = labels[sort_dist_indicies[i]]
        # class_count的值为label出现的次数
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 比较出现的概率，从大到小逆序排列
    # sort_class_count 被转化成一个元类的数组
    # sorted：operator.itemgetter(1)比较第一维大小（即出现次数）进行排序
    sort_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sort_class_count[0][0]


group, label = create_data_set()
print classify0([2, 2], group, label, 3)
