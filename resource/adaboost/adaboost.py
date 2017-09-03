#coding=utf-8

# 关于adaboost的文档
# http://www.360doc.com/content/14/1109/12/20290918_423780183.shtml


from numpy import *

def loadSimpData():
    dataMat=matrix([[1. ,2.1],
        [2. ,1.1],
        [1.3,1. ],
        [1. ,1. ],
        [2. ,1. ]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    result_array = ones((shape(dataMatrix)[0],1))
    if(threshIneq == 'lt'):
        result_array[dataMatrix[:,dimen] <= threshVal] = -1
    else:
        result_array[dataMatrix[:,dimen] > threshVal] = -1
    return result_array


def buildStump(dataArray, classLables, D):
    dataMatrix = mat(dataArray)
    labelMatrix = mat(classLables).T
    # 步数
    stepNums = 10.0
    bestStump = {}
    m,n = shape(dataMatrix)
    # 最佳预测
    bestClassLabelEst = mat(ones((m, 1)))
    # 最小误差
    minError = inf
    # 遍历所有特征
    for i in range(n):
        f_max = dataMatrix[:, i].max()
        f_min = dataMatrix[:, i].min()
        # 步长
        stepSize = (f_max - f_min) / stepNums
        for j in range(-1, int(stepNums) + 1):
            # 阈值符号(小于或者大于)
            for threshIneq in ['lt','gt']:
                # 阈值
                threshVal = f_min + float(j) * stepSize
                predictVal = stumpClassify(dataMatrix, i, threshVal, threshIneq)
                errorArray = mat(ones((m, 1)))
                errorArray[predictVal == labelMatrix] = 0
                error = D.T * errorArray
                # print 'error:', error
                if error < minError:
                    minError = error
                    bestStump['feature_index'] = i
                    bestStump['threshIneq'] = threshIneq
                    bestStump['threshVal'] = threshVal
                    bestClassLabelEst = predictVal
    # 输出决策树、错误率、样本估值
    return bestStump, minError, bestClassLabelEst


def adaBoostTrain(dataArray, classLables, iterator = 40):
    weakClassArray = []
    m = shape(dataArray)[0]
    # D是每个样本的权重矩阵，初始化每个样本权重等同
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(iterator):
        print "iterator:", i
        labelsMatT = mat(classLables).T
        bestStump, minError, bestClassLabelEst = buildStump(dataArray, classLables, D)
        print "D:", D.T
        print "bestClassLabelEst", bestClassLabelEst
        alpha = float(0.5 * log((1.0 - minError) / (max(minError, 1e-16))))
        bestStump['alpha'] = alpha
        weakClassArray.append(bestStump)
        expon = multiply(-1 * alpha * labelsMatT, bestClassLabelEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * bestClassLabelEst
        print("aggClassEst",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != labelsMatT, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print 'total error:', errorRate, '\n'
        if errorRate == 0.0:
            break
    return weakClassArray
        

dataSet, labels = loadSimpData()
# D = mat(ones((5,1)) / 5)
# print buildStump(dataSet, labels, D)

print adaBoostTrain(dataSet, labels, 9)
