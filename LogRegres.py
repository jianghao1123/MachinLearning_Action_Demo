# coding=utf-8


from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('resource/testSet.txt')
    for line in fr.readlines():
        lineArray = line.strip().split()
        dataMat.append([1.0, float(lineArray[0]), float(lineArray[1])])
        labelMat.append(int(lineArray[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 梯度上升
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    dataMatrixTranspose = dataMatrix.transpose()
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights += alpha * dataMatrixTranspose * error
    return weights


# 随机梯度上升
def stocGradAscent(dataMatIn, labelMat):
    dataMatrix = array(dataMatIn)
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(dataMatrix[i] * weights)
        error = labelMat[i] - h
        weights += alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatIn, classLabels, numIter=150):
    dataMatrix = array(dataMatIn)
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights += alpha * error * dataMatrix[i]
            del(dataIndex[randIndex])
    return weights




dataArr, labelMat = loadDataSet()
print stocGradAscent1(dataArr, labelMat)