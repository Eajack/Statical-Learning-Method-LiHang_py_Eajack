'''
!/usr/bin/env python3
-*- coding : utf-8 -*-
Author: Eajack
date:2018/12/12
Function：
    《统计学习方法-李航》 第四章
       1- 朴素贝叶斯算法 实现（对应极大似然估计）
       2- 贝叶斯估计（拉普拉斯平滑） 实现
'''
import numpy as np

def naiveBayes(train_X_list, train_Y_list, test_X, bayes_estimation_lambda=0):
    '''

    :param train_X_list: 输入X点list，X_i 为N维向量
    :param train_Y_list: 输入X点对应的分类标签Y
    :param test_X: 测试X点，N维向量
    :param bayes_estimation_lambda: 贝叶斯估计，对朴素贝叶斯的修正，引入的lambda参数
                                    bayes_estimation_lambda = 0 ==> naive bayes;
                                    bayes_estimation_lambda > 0 ==> bayes estimation
    :return: label_X: test_X对应类别
    Funtion:
        0- 前提：train_X_list中的 X_i，对应的每个维度的X独立分布，即“朴素”（naive）的含义

                                        P(X=x;Y=y_i)           P(X=x|Y=y_i) * P(Y=y_i)
        1- naive bayes：P(Y=y_i|X=x) = -------------- = --------------------------------------
                                           P(X=x)                       P(X=x)

                                                   
        2- bayes estimation：
                                    P(X=x|Y=y_i) * P(Y=y_i) + lambda
            P(X=x|Y=y_i) = --------------------------------------------------
                                                P(Y=y_i)

                            SIGMA-N_i [I(y=y_i)] + lambda
            P(Y=y_i) = --------------------------------------------------
                                        N + K*lambda

        3- naive bayes & bayes estimation: 实际中test_X概率，即P(X=x)，可以不用计算，因为是定值。因此，算法只需要计算分子即可。

        4- bayes_estimation_lambda = 1，拉普拉斯平滑(Laplace smoothing)

        5- naive bayes & bayes estimation差别在于: P(X=x|Y=y_i) & P(Y=y_i)的计算公式不一致

    '''
    if(len(train_X_list) != len(train_Y_list)):
        print("训练样本&标签个数，不一致！！！")
        return
    #1- 计算 P(Y=y_i) 所有概率
    trainNum = len(train_Y_list)
    trainY_kinds_list = list(set(train_Y_list))
    K = len(trainY_kinds_list)  #贝叶斯估计K，《统计学习方法-李航》 P51, 4.11公式
    P_Yy_j_list = [ (train_Y_list.count(y_j)+bayes_estimation_lambda)/(trainNum+K*bayes_estimation_lambda) \
        for y_j in trainY_kinds_list ]
    trainY_Num_list = [ train_Y_list.count(y_j) for y_j in trainY_kinds_list ]

    #2- 计算P(X=x|Y=y_i)
    P_XY = [0] * len(train_X_list[0])
    train_X_kinds_list = []  # 统计X种类，train_X_kinds_list[i]代表，点X的第i维度的值类别

    #2.1 对N维X分别计算 X_i 与 Y的联合概率表（二维矩阵）
    for X_count in range(len(P_XY)):
        train_Xi_values_list = [ train_X[X_count] for train_X in train_X_list ]
        train_Xi_kinds_list = list(set(train_Xi_values_list))
        train_X_kinds_list.append(train_Xi_kinds_list)
        P_XiY = np.zeros(shape=(len(train_Xi_kinds_list), len(trainY_kinds_list)))
        S_j = len(train_Xi_kinds_list)  #贝叶斯估计S_j，《统计学习方法-李航》 P51, 4.10公式

        # 2.2 遍历计算 p_xrow_ycol
        for row in range(len(P_XiY)):
            for col in range(len(P_XiY[0])):
                X_i = train_Xi_kinds_list[row]
                Y_i = trainY_kinds_list[col]
                XiYi_count = 0
                for index in range(len(train_Y_list)):
                    if( train_Y_list[index]==Y_i and train_Xi_values_list[index]==X_i ):
                        XiYi_count += 1
                p_Xi_Yi = (XiYi_count+bayes_estimation_lambda) / (trainY_Num_list[col]+S_j*bayes_estimation_lambda)

                P_XiY[row][col] = p_Xi_Yi

        P_XY[X_count] = P_XiY

    #3- 计算 P(X=x|Y=y_i) * P(Y=y_i)
    P_Yjs = [0] * len(trainY_kinds_list)
    for trainY_count in range(len(P_Yjs)):
        P_trainY = P_Yy_j_list[trainY_count]
        #3.2 计算P(X=x|Y=y_i),即 P(X=x|Y=trainY_kinds_list[trainY_count])
        P_X_Ynow = 1
        for testX_dimCount in range(len(test_X)):
            test_X_i = test_X[testX_dimCount]
            try:
                test_X_i_index = train_X_kinds_list[testX_dimCount].index(test_X_i)
                P_X_Ynow *= P_XY[testX_dimCount][test_X_i_index][trainY_count]
            except ValueError:
                # 训练集中没有test_X_i，即测试集test_X中存在某一维度的数值，在训练集概率为0
                P_X_Ynow *= 1

        #3.3 结果
        P_Yj = P_trainY*P_X_Ynow
        print("\t第{}个Y标签({})，概率为：{}".format(trainY_count+1, trainY_kinds_list[trainY_count], P_Yj))
        P_Yjs[trainY_count] = P_Yj

    #4- 结果输出
    test_X_label = trainY_kinds_list[P_Yjs.index(max(P_Yjs))]
    return test_X_label


if __name__ == '__main__':
    #0- 数据集准备
    # 用的是《统计学习方法-李航》 P50页例4.1数据
    train_X_list = [[1,'S'], [1,'M'], [1,'M'], [1,'S'], [1,'S'], \
                    [2,'S'], [2,'M'], [2,'M'], [2,'L'], [2,'L'], \
                    [3,'L'], [3,'M'], [3,'M'], [3,'L'], [3,'L'] ]
    train_Y_list = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    test_X = [2,'S']

    #1- 朴素贝叶斯算法
    print("1- 朴素贝叶斯算法")
    test_X_label = naiveBayes(train_X_list, train_Y_list, test_X, 0)
    print("\ttest_X 分类标签: {}".format(test_X_label))
    
    #2- 贝叶斯估计（拉普拉斯平滑）
    print("2- 贝叶斯估计（拉普拉斯平滑）")
    test_X_label = naiveBayes(train_X_list, train_Y_list, test_X, 1)
    print("\ttest_X 分类标签: {}".format(test_X_label))