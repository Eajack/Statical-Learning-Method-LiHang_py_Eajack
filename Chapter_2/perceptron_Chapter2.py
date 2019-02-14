#!/usr/bin/env python3
# -*- coding : utf-8 -*-
# Author: Eajack
# date:2018/12/8
# Function：
#    《统计学习方法-李航》 第二章
#       1- perceptron: 感知机算法

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

def drawLines(w_list, b, X_list, title_str):
    '''
    Funtion:
        画二维直线, f(x) = w_list[0]*x + w_list[1]*x + b
    :param w_list: 
    :param b: 
    '''
    ## X/Y散点图
    X_list_x = [item[0] for item in X_list]
    X_list_y = [item[1] for item in X_list]
    X_value = min(X_list_x)
    X_value_list = []
    Y_value_list = []
    while(X_value <= max(X_list_x)):
        X_value_list.append(X_value)
        Y_value_list.append( (-w_list[0]*X_value-b)/w_list[1] )
        X_value += 0.01

    drawDots(X_list_x, X_list_y)
    plt.scatter(X_value_list, Y_value_list, marker = '.',color = 'green', s = 5)
    plt.title(title_str)

    plt.show()

def drawDots(X_list, Y_list):
    '''
    Funtion:
        画散点图
    :param X_list: 输入自变量X的list
    :param Y_list: 输入因变量Y的list
    '''
    ## X/Y散点图
    plt.scatter(X_list[:50], Y_list[:50], marker = 'x',color = 'red', s = 10 ,label = '-1')
    plt.scatter(X_list[50:100], Y_list[50:100], marker = 'o',color = 'blue', s = 10 ,label = '1')
    plt.legend()

def produceData():
    # 借助于load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['X', 'Y', 'petal length', 'petal width', 'label']
    df.label.value_counts()
    plt.scatter(df[:50]['X'], df[:50]['Y'], label='-1')
    plt.scatter(df[50:100]['X'], df[50:100]['Y'], label='1')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    X_list = [ [df[0:100]['X'][i],df[0:100]['Y'][i]] for i in range(0,100) ]
    Y_list = ([-1] * 50)
    Y_list.extend([1] * 50)
    #print(X_list, Y_list)
    return [X_list, Y_list]

def perceptron_traditional(X_list, Y_list, yita = 0.1, maxLoopTime = 10000, drawFlag=1):
    '''
    Funtion:
        感知机算法，原始形式, SGD算法（随机梯度下降，stochastic Gradient Descent）
    :param X_list: 输入自变量X的list
    :param Y_list: 输入因变量Y的list
    :param yita: 学习率，[0,1]
    :param maxLoopTime: 最大迭代次数，防止永远while
    :return: [w_list,b,loopCount] => 感知机模型: f(x) = sign(w_list.x + b)
    '''
    #1- 初始化w, b
    w_list = np.array([0] * len(X_list[0]), dtype=np.float64)
    b = 0
    X_list = np.array(X_list, dtype=np.float64)
    Y_list = np.array(Y_list, dtype=np.float64)

    #2- 遍历X_list, Y_list
    loopCount = 0
    breakFlag = 0
    while(breakFlag != 1):
        loopCount += 1
        if(loopCount >= maxLoopTime):
            break

        passCount = 0
        for pointCount in range(0,len(X_list)):
            featureValue = Y_list[pointCount] * (np.vdot(w_list, X_list[pointCount]) + b)
            if(featureValue <= 0):
                w_list += (yita*Y_list[pointCount]*X_list[pointCount])
                b += (yita*Y_list[pointCount])
                #print("第" + str(loopCount) + "次迭代: ", w_list, b, breakFlag)
                breakFlag = 0
                break
            else:
                passCount += 1

            if(passCount == len(X_list)):
                breakFlag = 1
                break

    w_list = list(w_list)

    #画图
    if(drawFlag):
        print('perceptron_traditional model 迭代次数：' + str(loopCount))
        drawLines(w_list, b, X_list, 'perceptron_traditional model')

    return [w_list,b,loopCount]

def perceptron_dualForm(X_list, Y_list, yita = 0.1, maxLoopTime = 10000, drawFlag=1):
    '''
    Funtion:
        感知机算法，对偶形式, SGD算法（随机梯度下降，stochastic Gradient Descent）
    :param X_list: 输入自变量X的list
    :param Y_list: 输入因变量Y的list
    :param yita: 学习率，[0,1]
    :param maxLoopTime: 最大迭代次数，防止永远while
    :param drawFlag: 1 => 画图
    :return: [alpha,b] => f(x) = sign[(sigma-N_j=1)(alpha_j * y_j * x_j).x + b)]
                最终 w_list,b,loopCount
    '''
    #1- 初始化alpha, b
    alpha = np.array([0] * len(X_list), dtype=np.float64)
    b = 0
    X_list = np.array(X_list, dtype=np.float64)
    Y_list = np.array(Y_list, dtype=np.float64)
    N = len(alpha)

    #2- 遍历X_list, Y_list
    loopCount = 0
    breakFlag = 0
    while(breakFlag != 1):
        loopCount += 1
        if(loopCount >= maxLoopTime):
            break

        passCount = 0
        for pointCount in range(0,len(X_list)):
            # 计算 featureValue
            bufferVector = 0
            for i in range(0,N):
                bufferVector += (alpha[i]*Y_list[i]*X_list[i])
            bufferVector = np.array(bufferVector)
            featureValue = Y_list[pointCount] * (np.vdot(bufferVector,X_list[pointCount]) + b)

            if(featureValue <= 0):
                alpha[pointCount] += (yita)
                b += (yita*Y_list[pointCount])
                #print("第" + str(loopCount) + "次迭代: ", alpha, b, breakFlag)
                breakFlag = 0
                break
            else:
                passCount += 1

            if(passCount == len(X_list)):
                breakFlag = 1
                break

    #3- 计算w, b返回
    w_list = 0
    for i in range(0,N):
        w_list += (alpha[i]*Y_list[i]*X_list[i])
    w_list = list(w_list)
    
    #画图
    if(drawFlag):
        print('perceptron_dualForm model 迭代次数：' + str(loopCount))
        drawLines(w_list, b, X_list, 'perceptron_dualForm model')

    return [w_list,b,loopCount]


if __name__ == '__main__':
    #0- 数据集准备
    [X_list, Y_list] = produceData()

    #1- 感知机 测试
    # x_positive_1 = [3,3]
    # x_positive_2 = [4,3]
    # x_negative_1 = [1,1]
    # X_list = [x_positive_1, x_positive_2, x_negative_1]
    # Y_list = [1, 1, -1]

    w_b_loopCount_list_1 = perceptron_traditional(X_list, Y_list, 0.2, 10000, 1)
    w_b_loopCount_list_2 = perceptron_dualForm(X_list, Y_list, 0.2, 10000, 1)

    #2- 学习率 & 迭代次数 曲线图
    yita = 0
    yita_list = []
    perceptron_traditional_loopCount_list = []
    perceptron_dualForm_loopCount_list = []
    while(yita < 1):
        yita_list.append(yita)
        trash_list = perceptron_traditional(X_list, Y_list, yita, 10000, 0)
        perceptron_traditional_loopCount_list.append(trash_list[2])

        trash_list = perceptron_dualForm(X_list, Y_list, yita, 10000, 0)
        perceptron_dualForm_loopCount_list.append(trash_list[2])

        yita += 0.01

    plt.plot(yita_list, perceptron_traditional_loopCount_list)
    plt.title('traditional-yita Curve')
    plt.show()

    plt.plot(yita_list, perceptron_dualForm_loopCount_list)
    plt.title('dualForm-yita Curve')
    plt.show()
