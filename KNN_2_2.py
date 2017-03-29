# coding=utf-8


from numpy import *
import  matplotlib
import matplotlib.pyplot as plt
import KNN_2_1


# 从文件中读取矩阵
def file2matrix(filename):
    fr = open(filename)
    array_lines = fr.readlines()
    lines_length = len(array_lines)
    # 创建一个lines_length行3列的矩阵
    data = zeros((lines_length, 3))
    label_array = []
    index = 0
    for line in array_lines:
        line = line.strip()
        line_item_list = line.split('\t')
        # 前三个数据存入矩阵
        # data[index,:]表示取第index行，data[:,index]表示取第index列的数据
        data[index, :] = line_item_list[0:3]
        # 最后一个存入分类数组
        label_array.append(int(line_item_list[-1]))
        index += 1
    return data, label_array


# 归一化数值，new = (old - min) / (max - min)
def auto_norm(data):
    min = data.min(0)
    max = data.max(0)
    diff = max - min
    m = data.shape[0]
    norm_data = data - tile(min, (m, 1))
    norm_data /= tile(diff, (m, 1))
    return norm_data, diff, min


def dating_test():
    # 测试数据比例
    ho_ratio = 0.10
    data, labels = file2matrix('resource/datingTestSet.txt')
    norm_data, diff, min = auto_norm(data)
    m = norm_data.shape[0]
    # 测试数据数量
    test_count = int(m * ho_ratio)
    # 错误数量
    error_count = 0.0
    for i in range(test_count):
        label_result = KNN_2_1.classify0(norm_data[i, :], norm_data[test_count:m, :], labels[test_count:m], 4)
        print u'预测分类%d,实际分类%d' % (label_result, labels[i])
        if label_result != labels[i]:
            error_count += 1.0
    print u'预测错误率%f' % (error_count / float(test_count))


def classify_person():
    while True:
        result = [u'不感兴趣', u'感兴趣', u'喜欢']
        miles = float(raw_input(u'每年获得的飞行常客里程数'))
        games = float(raw_input(u'输入玩视频游戏时间百分比'))
        ice = float(raw_input(u'每周消费的冰激凌公升数'))
        data, labels = file2matrix('resource/datingTestSet.txt')
        norm_data, diff, min = auto_norm(data)
        in_array = array([miles, games, ice])
        label_result = KNN_2_1.classify0((in_array - min) / diff, norm_data, labels, 4)
        print u'你可能对这个人', result[label_result - 1]

# dating_test()
classify_person()


# data, label_array = file2matrix('resource/datingTestSet.txt')
# norm_data, diff, m = auto_norm(data)
# print norm_data
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(data[:, 1], data[:, 2], 15.0 * array(label_array), 15.0 * array(label_array))
# plt.show()
