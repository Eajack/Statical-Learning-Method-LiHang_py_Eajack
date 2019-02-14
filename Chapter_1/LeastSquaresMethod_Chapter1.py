#!/usr/bin/env python3
# -*- coding : utf-8 -*-
# Author: Eajack
# date:2018/11/11
# Function：
#    《统计学习方法-李航》 第一章
#       1- LSM： least square method（最小二乘法）
#       2- LSM拟合 y = sin(2*pi*x) (P11-例1)
#       3- 正则化项引入: L2范数；L1没找到解法
# 补坑：
#    1- 实现L1范数正则化 + LSM

import matplotlib.pyplot as plt
import numpy as np

def drawDots(X_list, Y_list):
    '''
    Funtion:
        画散点图
    :param X_list: 输入自变量X的list
    :param Y_list: 输入因变量Y的list
    '''
    ## X/Y散点图
    plt.scatter(X_list, Y_list, marker = 'x',color = 'red', s = 60 ,label = 'real dots')
    plt.show()

def realFunction(X_list):
    '''
    Funtion:
        目标函数，y = sin(2*pi*x)
    :param X_list: 输入自变量X的list
    :param Y_list: 输入因变量Y的list
    :return: Y_list
    '''
    # 同理，可换成其他函数，LSM很牛逼。此处为 y = sin(2*pi*x)
    return [ np.sin(2*np.pi*x) for x in X_list ]

def LSM_MultiX(X_list, Y_list, k):
    '''
    Funtion:
        多项式拟合，等价于多元线性回归，最小二乘法(Least Square Method)
        Y = W * X (向量相乘)
        eg: y = x^4 + x^3 + x^2 + x + 1
    :param X_list: 输入自变量X的list
    :param Y_list: 输入因变量Y的list
    :param k: k阶多项式，eg: k = 2 => x^2 + x + 1
    :return: [w]
    '''
    ## 参照：
    #   1- 机器学习-周志华, P55
    #   2- http://littleshi.cn/online/LstSqr.html
    # 最重要公式： W = inv(X'*X)*X'*Y
    # 算法复杂度来源： 1次求逆，1次转置、3次矩阵相乘
    n = len(X_list)
    X_matrix = [0] * (n*(k+1))
    listCount = 0
    for rowCount in range(0,n):
        for colCount in range(0,k+1):
            if(colCount == k+1):
                X_matrix[listCount] = 1
            else:
                X_matrix[listCount] = np.power(X_list[rowCount], colCount)
            listCount += 1
    
    # W = inv(X'*X)*X'*Y
    X_matrix = np.mat(np.array(X_matrix).reshape(n,k+1))
    Y_matrix = np.mat(np.array(Y_list).reshape(-1,1))   #转成列向量
    X_matrix_T = X_matrix.transpose() #转置

    W = (np.linalg.inv(X_matrix_T * X_matrix)) *  X_matrix_T * Y_matrix
    W = np.matrix.tolist(W)
    W = [ w[0] for w in W ]
    W.reverse()
    #print(W)

    # 多项式曲线，画图
    # 多项式
    polyFunction = np.poly1d(W)
    print(polyFunction)
    x_index = min(X_list)
    x_index_new = []
    y_index = []
    while(x_index <= max(X_list)):
        x_index_new.append(x_index)
        x_index += 0.01
        y_index.append(polyFunction(x_index))
        #plt.scatter(x_index, polyFunction(x_index), marker = '.',color = 'blue', s = 60 ,label = 'LSM' + str(k))
    
    plt.scatter(X_list, Y_list, marker = 'x',color = 'red', s = 60 ,label = 'real dots')
    plt.plot(x_index_new, y_index, label = 'LSM-' + str(k))
    plt.legend()
    plt.show()

    return W

def LSM_witRegulrization(X_list, Y_list, k, L, regulrize_lambda = 0.0001):
    '''
    Funtion:
        最小二乘法(Least Square Method)，配上正则化项
        L1 / L2 范数项
    :param L: L=1 => L1范数正则、L=2 => L2范数正则
    :param regulrize_lamda: 正则化lambda系数。值太小，过拟合；值太大，欠拟合
    以下与 LSM_MultiX 一致
    :param X_list:
    :param Y_list:
    :param k:
    :return: W
    '''
    # 复用 LSM_MultiX 函数
    ## 参照：
    #   1- 统计学习方法- P13-14
    #   2- L2防止过拟合：https://blog.csdn.net/chunyun0716/article/details/50812416
    #   3- L1暂时没找到，《机器学习-周志华》P253-254有介绍，暂时没看懂
    #   4- https://blog.csdn.net/dou3516/article/details/78795721
    # 最重要公式：
    #   1- L1正则化 => W = (np.linalg.inv(X_matrix_T * X_matrix)) * (X_matrix_T * Y_matrix - regulrize_lambda * 0.5)
    #   2- L2正则化 => W = inv(X'*X + regulrize_lambda * I) * X'*Y
    n = len(X_list)
    X_matrix = [0] * (n*(k+1))
    listCount = 0
    for rowCount in range(0,n):
        for colCount in range(0,k+1):
            if(colCount == k+1):
                X_matrix[listCount] = 1
            else:
                X_matrix[listCount] = np.power(X_list[rowCount], colCount)
            listCount += 1
    
    # W = inv(X'*X)*X'*Y
    X_matrix = np.mat(np.array(X_matrix).reshape(n,k+1))
    Y_matrix = np.mat(np.array(Y_list).reshape(-1,1))   #转成列向量
    X_matrix_T = X_matrix.transpose() #转置

    if(L == 1):
        # L1
        # Failed...
        W = (np.linalg.inv(X_matrix_T * X_matrix)) * (X_matrix_T * Y_matrix - regulrize_lambda * 0.5)
    elif(L == 2):
        # L2
        W = (np.linalg.inv(X_matrix_T*X_matrix + regulrize_lambda * np.eye(k+1))) *  X_matrix_T * Y_matrix
    else:
        #默认无正则
        W = (np.linalg.inv(X_matrix_T * X_matrix)) *  X_matrix_T * Y_matrix

    W = np.matrix.tolist(W)
    W = [ w[0] for w in W ]
    W.reverse()
    #print(W)

    # 多项式曲线，画图
    # 多项式
    polyFunction = np.poly1d(W)
    print(polyFunction)
    x_index = min(X_list)
    x_index_new = []
    y_index = []
    while(x_index <= max(X_list)):
        x_index_new.append(x_index)
        x_index += 0.01
        y_index.append(polyFunction(x_index))
        #plt.scatter(x_index, polyFunction(x_index), marker = '.',color = 'blue', s = 60 ,label = 'LSM' + str(k))
    
    plt.scatter(X_list, Y_list, marker = 'x',color = 'red', s = 60 ,label = 'real dots')
    plt.plot(x_index_new, y_index, label = 'LSM-Regularize-' + str(k))
    plt.legend()
    plt.show()

    return W

if __name__ == '__main__':
    #1- 20个点，加入正态分布噪声
    X_list = np.linspace(0, 1, 20)
    Y_list = realFunction(X_list)
    Y_list_addNoise = [ np.random.normal(0, 0.1)+y for y in Y_list ]

    #2- 画散点图
    drawDots(X_list, Y_list_addNoise)

    #3- LSM
    LSM_MultiX(X_list, Y_list_addNoise, 4)  #无 过拟合
    LSM_MultiX(X_list, Y_list_addNoise, 17) #出现 过拟合

    #4- 正则化
    # 对k = 17时，加入L1 / L2 正则化项
    # L1作用： 使得数据稀疏起来，即制造更多0；缓解过拟合
    # L2作用： 可以显著缓解过拟合
    LSM_witRegulrization(X_list, Y_list_addNoise, 17, 1, 0.0001)
    LSM_witRegulrization(X_list, Y_list_addNoise, 17, 2, 0.0001)
