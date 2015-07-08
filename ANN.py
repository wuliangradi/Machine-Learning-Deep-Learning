#coding=utf-8

'''
datetime: 2015-7-2 12:43
author: Wuliang
'''

import numpy as np
import matplotlib.pyplot as plt
import struct
import math
import random
import time

def sigmoid(inX):
    '''
    激活函数 得到 hypothesis function 假设函数
    :param inX:
    :return:
    '''
    return 1.0/(1.0 + math.exp(-inX))

def tangent(inX):
    '''
    激活函数之”双曲正切函数“
    :param inX:
    :return:
    '''
    return (1.0*math.exp(inX) - 1.0*math.exp(-inX)) / (1.0*math.exp(inX) + 1.0*math.exp(-inX))

def softmax(inMatrix):
    '''
    softmax函数
    :param inMatrix:
    :return:
    '''
    m, n = np.shape(inMatrix)
    outMatrix = np.mat(np.zeros((m, n)))
    soft_sum = 0
    for idx in range(0, n):
        outMatrix[0, idx] = math.exp(inMatrix[0, idx])
        soft_sum += outMatrix[0, idx]
    for idx in range(0, n):
        outMatrix[0, idx] /= soft_sum
    return outMatrix

def difsigmoid(inX):
    '''
    softmax函数的导数
    :param inX:
    :return:
    '''
    return sigmoid(inX) * (1.0-sigmoid(inX))

def sigmoidMatrix(inputMatrix):
    '''
    计算矩阵的激活值
    :param inputMatrix:
    :return:
    '''
    m, n = np.shape(inputMatrix)
    outMatrix = np.mat(np.zeros((m, n)))
    for idx_m in range(0, m):
        for idx_n in range(0, n):
            outMatrix[idx_m, idx_n] = sigmoid(inputMatrix[idx_m, idx_n])
    return outMatrix

def loadMNISTimage(filePathName, dataNum = 60000):
     images = open(filePathName, 'rb')
     buf = images.read()
     index = 0
     magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
     index += struct.calcsize('>IIII')
     if magic != 2051:
         raise Exception
     datasize = int(784*dataNum)
     datablock = ">" + str(datasize) + "B"

     nextmatrix = struct.unpack_from(datablock, buf, index)
     nextmatrix = np.array(nextmatrix) / 255.0
     nextmatrix = nextmatrix.reshape(dataNum, 1, numRows*numColumns)
     return nextmatrix, numImages

def loadMNISTlabels(filePathName, dataNum = 60000):
     labels = open(filePathName, 'rb')
     buf = labels.read()
     index = 0
     magic, numLabels = struct.unpack_from(">II", buf, index)
     index += struct.calcsize('>II')
     if magic != 2049:
         raise Exception
     datablock = ">" + str(dataNum) + "B"
     nextmatrix = struct.unpack_from(datablock, buf, index)
     nextmatrix = np.array(nextmatrix)
     return nextmatrix, numLabels

class MuiltilayerANN(object):
    def __init__(self, numOfHiddenLayers, numOfNodesInHiddenLayers, inputDimension, outDimension, maxIter=50):
        self.trainDataNum = 200                       # 训练数据样本数
        self.decayRate = 0.2                          # 学习步长
        self.punishFactor = 0.05                      # 惩罚因子
        self.eps = 0.00001                            # 误差阈值
        self.numOfHL = numOfHiddenLayers              # 隐藏层层数
        self.NL = int(numOfHiddenLayers + 2)          # 网络层数
        self.nodesInHidden = []
        for element in numOfNodesInHiddenLayers:
            self.nodesInHidden.append(int(element))   # 隐藏层每层的节点数
        self.inputDi = int(inputDimension)            # 输入层维数
        self.outDi = int(outDimension)                # 输出层维数
        self.maxIteration = int(maxIter)              # 最大迭代次数
        
    def setTrainDataNum(self, dataNum):
        '''
        设置训练样本数
        :param dataNum:
        :return:
        '''
        self.trainDataNum = dataNum
        return

    def loadTrainData(self, filePath):
        '''
        导入训练样本数据
        :param filePath:
        :return:
        '''
        self.trainData, self.numTrainData = loadMNISTimage(filePath, self.trainDataNum)
        return

    def loadTrainLabel(self, filePath):
        '''
        导入训练标签数据
        :param filePath:
        :return:
        '''
        self.trainLabel, self.numTrainLabel = loadMNISTlabels(filePath, self.trainDataNum)
        if self.numTrainData != self.numTrainLabel:
            raise Exception
        return

    def initialWeights(self):
        '''
        随机初始化权重矩阵
        :return:
        '''
        self.nodesInLayers = []
        self.nodesInLayers.append(int(self.inputDi))
        self.nodesInLayers += self.nodesInHidden    # 这个加法也是厉害啊
        self.nodesInLayers.append(int(self.outDi))
        self.weightMatrix = []      # 权值矩阵（是一个list, list元素为矩阵）连接线上的权值
        self.B = []                 # 偏移矩阵
        for idx in range(0, self.NL - 1):
            s = math.sqrt(6) / math.sqrt(self.nodesInLayers[idx] + self.nodesInLayers[idx+1])
            # 参考Adnerw NG机器学习课程
            # A good choice of ε(init) is ε(init) = sqrt(6) / sqrt( + ), where Lin = sl and Lout = sl+1
            tempMatric = np.zeros((self.nodesInLayers[idx], self.nodesInLayers[idx+1]))
            for row in range(self.nodesInLayers[idx]):
                for col in range(self.nodesInLayers[idx+1]):
                    tempMatric[row, col] = random.random() * 2.0 * s - s
            self.weightMatrix.append(np.mat(tempMatric))
            self.B.append(np.mat(np.zeros((1, self.nodesInLayers[idx+1]))))
        return 0

    def printWeightMatric(self):
        '''
        打印权重矩阵
        :return:
        '''
        for idx in range(int(self.NL) - 1):
            print "第", idx, "层神经网络到第", idx+1, "层神经网络"
            print "WeightMatrix is >>>> "
            print self.weightMatrix[idx]
            print "Intercept is    >>>> "
            print self.B[idx]
        return 0

    def forwardPropogation(self, singleDataInput, currentDataIdx):
        '''
        前向传播
        :param singleDataInput:
        :param currentDataIdx:
        :return:
        '''
        Ztemp = []
        Ztemp.append(np.mat(singleDataInput) * self.weightMatrix[0] + self.B[0])
        Atemp = []           # 激活值---activations

        for idx in range(1, self.NL-1):
            Atemp.append(sigmoidMatrix(Ztemp[idx-1]))
            Ztemp.append(Atemp[idx-1] * self.weightMatrix[idx] + self.B[idx])  # Wx + b
        Atemp.append(Ztemp[self.NL-2])      # h(x)值

        errorMat = softmax(Atemp[self.NL-2])
        errorsum = -1.0 * math.log(errorMat[0, int(self.trainLabel[currentDataIdx])])   # 总的代价值
        return Atemp, Ztemp, errorsum

    def calThetaNL(self, Anl, y, Znl):
        thetaNL = softmax(Anl) - y
        return thetaNL

    def backwardPropogation(self, singleDataInput, currentDataIdx):
        '''
        后向反馈
        :param singleDataInput:
        :param currentDataIdx:
        :return:
        '''
        Theta = []
        Atemp, Ztemp, tempError = self.forwardPropogation(np.mat(singleDataInput), currentDataIdx)
        outlabels = np.mat(np.zeros((1, self.outDi)))
        outlabels[0, int(self.trainLabel[currentDataIdx])] = 1.0
        # 首先outlabels是一个1*10的向量，是哪一个标签值，该标签值作为索引的数组值为1.0

        thetaNL = self.calThetaNL(Atemp[self.NL-2], outlabels, Ztemp[self.NL-2])  # 对输出层计算
        Theta.append(thetaNL)

        for idx in range(1, self.NL-1):
            inverseIdx = self.NL - 1 - idx
            thetaLPlus1 = Theta[idx-1]
            WeightL = self.weightMatrix[inverseIdx]
            ZL = Ztemp[inverseIdx - 1]
            theatL = thetaLPlus1 * WeightL.transpose()
            row_theta, col_theta = np.shape(theatL)
            if row_theta != 1:
                raise Exception
            for idx_col in range(col_theta):
                theatL[0, idx_col] = theatL[0, idx_col] * difsigmoid(ZL[0, idx_col])  # sigmoid函数的倒数
            Theta.append(theatL)

        DetaW = []
        DetaB = []
        for idx in range(self.NL-2):
            inverseIdx = self.NL-2-1-idx
            dW = Atemp[inverseIdx].transpose() * Theta[idx]
            dB = Theta[idx]
            DetaW.append(dW)
            DetaB.append(dB)
        DetaW.append(singleDataInput.transpose() * Theta[self.NL-2])
        DetaB.append(Theta[self.NL-2])
        return DetaW, DetaB, tempError

    def updatePara(self, DetaW, DetaB):
        '''
        更新权重矩阵
        :param DetaW:
        :param DetaB:
        :return:
        '''
        for idx in range(self.NL-1):
            inverseIdx = self.NL-1-1-idx
            self.weightMatrix[inverseIdx] -= self.decayRate*((1.0/self.trainDataNum) * DetaW[idx] +
                                             self.punishFactor * self.weightMatrix[inverseIdx])
            self.B[inverseIdx] -= self.decayRate * (1.0/self.trainDataNum) * DetaB[idx]

    def calPunish(self):
        '''
        计算惩罚项
        :return:
        '''
        puishment = 0.0
        for idx in range(self.NL-1):
            temp = self.weightMatrix[idx]
            idx_m, idx_n = np.shape(temp)
            for i_m in range(idx_m):
                for i_n in range(idx):
                    puishment += temp[i_m, i_n] * temp[i_m, i_n]
        return 0.5*self.punishFactor*puishment

    def trainANN(self):
        '''
        训练神经网络
        :return:
        '''
        Error_old = 10000000000.0
        iter_idx = 0
        while iter_idx < self.maxIteration:
            print  "Iter num: ", iter_idx, "*************************************"
            iter_idx +=1
            cDetaW, cDetaB, cError = self.backwardPropogation(self.trainData[0], 0)

            for idx in range(1, self.trainDataNum):
                DetaWtemp, DetaBtemp, Errortemp=self.backwardPropogation(self.trainData[idx],idx)
                cError += Errortemp
                for idx_W in range(0,len(cDetaW)):
                    cDetaW[idx_W] += DetaWtemp[idx_W]

                for idx_B in range(0,len(cDetaB)):
                    cDetaB[idx_B] += DetaBtemp[idx_B]

            cError /= self.trainDataNum
            cError += self.calPunish()
            print "old error", Error_old
            print "new error", cError
            Error_new = cError
            if Error_old - Error_new < self.eps:
                break
            Error_old = Error_new
            self.updatePara(cDetaW, cDetaB)
        return

    def getTrainAccuracy(self):
        '''
        训练精度
        :return:
        '''
        accuracycount=0
        for idx in range(0, self.trainDataNum):
            Atemp, Ztemp, errorsum = self.forwardPropogation(self.trainData[idx], idx)
            TrainPredict = softmax(Atemp[self.NL-2])
            Plist = TrainPredict.tolist()
            LabelPredict = Plist[0].index(max(Plist[0]))
            print "LabelPredict", LabelPredict
            print "trainLabel", self.trainLabel[idx]
            if int(LabelPredict) == int(self.trainLabel[idx]):
                accuracycount += 1
        print "accuracy:", float(accuracycount)/float(self.trainDataNum)
        return


if __name__=="__main__":
    Ann = MuiltilayerANN (1, [256], 784, 10, 200)
    Ann.setTrainDataNum(400)
    Ann.loadTrainData("/Users/wuliang/PycharmProjects/ml_all/DeepLearning/t10k-images-idx3-ubyte")
    Ann.loadTrainLabel("/Users/wuliang/PycharmProjects/ml_all/DeepLearning/t10k-labels-idx1-ubyte")
    Ann.initialWeights()
    Ann.trainANN()
    Ann.getTrainAccuracy()




















