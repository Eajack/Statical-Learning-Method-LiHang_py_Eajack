#!/usr/bin/env python3
# -*- coding : utf-8 -*-
# Author: Eajack
# date:2018/11/11
# Function：
#    《统计学习方法-李航》 第一章算法实现：一元线性回归，最小二乘法
# 补坑：

import matplotlib.pyplot as plt
import numpy as np

def LSM_SingleX(X_list, Y_list):
    '''
    Funtion:
        单变量线性回归，y = w*x + b，最小二乘法(Least Square Method)
    :param X_list: 输入自变量X的list
    :param Y_list: 输入因变量Y的list
    :return: [w,b]
    '''
    XY_num = len(X_list)
    X_average = np.mean(X_list)

    # 最小二乘法公式
    w = sum( np.array(Y_list)*(np.array(X_list) - X_average) ) / ( sum(np.power(X_list,2))-(1/XY_num)*np.power(sum(X_list),2) )
    b = (1/XY_num) * sum( np.array(Y_list) - w * np.array(X_list) )

    # 画图
    ## X/Y散点图
    plt.scatter(X_list, Y_list, marker = 'x',color = 'red', s = 60 ,label = 'First')
    ## X/Y回归直线
    x=np.linspace(min(X_list), max(X_list), 1000)
    y=[(w*i+b) for i in x]
    plt.plot(x,y)
    plt.show()

    return [w,b]

if __name__ == '__main__':
    X_list = [150, 200, 250, 300, 350, 400, 600]
    Y_list = [6450, 7450, 8450, 9450, 11450, 15450, 18450]
    return_W_b = LSM_SingleX(X_list, Y_list)
    print(return_W_b)