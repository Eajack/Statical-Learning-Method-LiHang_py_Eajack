#!/usr/bin/env python3
# -*- coding : utf-8 -*-
# Author: Eajack
# date:2018/12/10
# Function：
#    《统计学习方法-李航》 第三章
#       1- KNN 简单实现
#       2- kd树实现KNN，kd树有点难搞，感觉没完全懂，
#           参考了2个网址,代码直接修改了一个博客的……
# kd Tree的确牛逼的，才100个二维点，时间差距就这么大了
# ========================结果========================
# KNN_simple- test_X Label is: 1
#    -KNN_simple 用时： 0.13165112001913434 秒
# KNN_kdTree- test_X Label is: 1
#    -KNN_kdTree 用时： 0.008607168893740857 秒
# 时间是15倍啊！！！
# ========================结果========================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from collections import Counter
import time

class dataAndDraw(object):
    def __init__(self):
        '''

        :param
        '''
        pass
    def produceData(self):
        # 借助于load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['label'] = iris.target
        df.columns = ['X', 'Y', 'petal length', 'petal width', 'label']
        produce_X_list = [ [df[0:100]['X'][i],df[0:100]['Y'][i]] for i in range(0,100) ]
        produce_Y_list = ([-1] * 50)
        produce_Y_list.extend([1] * 50)
        #print(produce_X_list, produce_Y_list)
        return [produce_X_list, produce_Y_list]

    def drawDots(self, X_list, Y_list, marker, color, label):
        '''
        Funtion:
            画散点图
        :param X_list: 输入自变量X的list
        :param Y_list: 输入因变量Y的list
        '''
        ## X/Y散点图
        plt.scatter(X_list, Y_list, marker = marker,color = color, s = 30 ,label = label)
        plt.legend()

#1- KNN简单实现
class KNN_simple(object):
    def __init__(self, train_X_list, train_Y_list, test_X, k, p=2):
        '''

        :param train_X_list: 训练集点 X_i 为N维向量
        :param train_Y_list: 训练集 X_i 对应标签
        :param test_X: 测试点 X 为N维向量
        :param k: K参数
        :param p: L_p范数（距离对应p）,默认L2范数，即欧式距离
        :return: test_Y_list，测试集 X_i 标签
        '''
        #self.train_X_list = train_X_list
        self.train_Y_list = train_Y_list
        #self.test_X = test_X
        self.k = k
        self.p = p

        #归一化，对train_X_list & test_X 一起
        X_list_buffer = [train_X for train_X in train_X_list]
        X_list_buffer.append(test_X)

        for dim_count in range(0,len(X_list_buffer[0])):
            X_dim_buffer = [ item[dim_count] for item in X_list_buffer]
            X_dim_buffer_min = min(X_dim_buffer)
            X_dim_buffer_max = max(X_dim_buffer)
            X_dim_buffer_new = [ (np.array(X_point)-X_dim_buffer_min)/(X_dim_buffer_min-X_dim_buffer_max) for X_point in X_dim_buffer]

            for X_point_Count in range(len(X_list_buffer)):
                X_list_buffer[X_point_Count][dim_count] = X_dim_buffer_new[X_point_Count]


        self.train_X_list = X_list_buffer[0:-1]
        self.test_X = X_list_buffer[-1]

    def L_p(self, X_i, X_j):
        '''

        :param X_i:
        :param X_j:
        :return: X_i & X_j 的L_p距离
        '''
        return np.power( sum(np.power(abs(np.array(X_i) - np.array(X_j)), self.p)), (1/self.p) )
    
    def predict(self):
        #1- 遍历计算test_X & train_X_list所有点距离
        test_X_dist_list = []
        for train_X in self.train_X_list:
            test_X_dist_list.append(self.L_p(train_X, self.test_X))

        #2- test_X_dist_list倒序排列，取出前k个对应编号的 train_X_list的train_Y_list
        # 倒序排序
        test_X_distSorted_list = [ [test_X_dist_list[i],i] for i in range(len(test_X_dist_list)) ]
        test_X_distSorted_list.sort()

        # 前k个item
        test_X_dist_k_list = test_X_distSorted_list[0:self.k]
        # 前k个标签
        test_X_dist_K_index_list = [ item[1] for item in test_X_dist_k_list ]
        test_X_dist_K_labelY_list = [ self.train_Y_list[index] for index in test_X_dist_K_index_list]

        #3- 统计 test_X_dist_K_labelY_list 出现次数最多的标签
        # 分类决策：选择最多类
        # 注意：如果最多类别统计有多个，随机挑一个，此处只用Counter随机挑
        predictLabel = Counter(test_X_dist_K_labelY_list).most_common(1)[0][0]

        #4- draw
        dataAndDraw_fun = dataAndDraw()
        drawPoints_X = [ X[0] for X in self.train_X_list ]
        drawPoints_Y = [ X[1] for X in self.train_X_list ]

        dataAndDraw_fun.drawDots(drawPoints_X[:50], drawPoints_Y[:50], '*', 'red', '-1')
        dataAndDraw_fun.drawDots(drawPoints_X[50:100], drawPoints_Y[50:100], 'o', 'blue', '+1')
        dataAndDraw_fun.drawDots(self.test_X[0], self.test_X[1], 'x', 'green', 'test_X')
        plt.title('KNN_simple(normalization)')
        plt.show()

        return predictLabel

#2- kd树实现 KNN
# 参考：1- 《统计学习方法》 P41-44
#       2- https://blog.csdn.net/v_JULY_v/article/details/8203674 
#       3- 参考代码来源：https://www.jianshu.com/p/521f00393504
class kdNode(object):
    """docstring for kdNode"""
    '''
    定义KD节点：
    point:节点里面的样本点，指的就是一个样本点
    split:分割维度（即用哪个维度的数据进行切分，比如4维数据，split=3，则表示按照第4列的数据进行切分空间）
    left:节点的左子节点
    right:节点的右子节点
    '''
    def __init__(self, point, split, left, right):
        self.point = point
        self.split = split
        self.left = left
        self.right = right

class kdTree(object):
    def __init__(self, train_X_list):
        def create_kdNode(split, X_list):
            '''

            :param X_list: 输入N维空间点 list
            :param split: 分割维度
            :return: kdNode
            '''
            if len(X_list) == 0:
                return None
            X_list = list(X_list)
            # 按照split维度数据大小，进行拍排序
            X_list.sort(key=lambda x  : x[split])
            X_list = np.array(X_list)
            # 取中位数index
            median = len(X_list) // 2
            
            return kdNode(X_list[median], split, create_kdNode(maxVar(X_list[:median]),X_list[:median]), \
                create_kdNode(maxVar(X_list[median+1:]),X_list[median+1:]))
            
        def maxVar(X_list):
            '''

            :param X_list: 输入N维空间点 list
            :return: maxVar_index, 最大方差的维度，作为create_kdNode的split
            '''
            if len(X_list) == 0:
                return 0
            Xpoint_var_list = []
            for dim_count in range(len(X_list[0])):
                x_dimI_list = [ x_point[dim_count] for x_point in X_list]
                Xpoint_var_list.append(np.var(np.array(x_dimI_list)))
            maxVar_index = Xpoint_var_list.index(max(Xpoint_var_list))
            return maxVar_index
        
        # 根节点
        self.root = create_kdNode(maxVar(train_X_list),train_X_list)

def computeDist(pt1, pt2):
    """
    计算两个数据点的距离
    return:pt1和pt2之间的距离
    """
    sum = 0.0
    for i in range(len(pt1)):
        sum = sum + (pt1[i] - pt2[i]) * (pt1[i] - pt2[i])
    return np.math.sqrt(sum)

def updateNN(min_dist_array=None, tmp_dist=0.0, NN=None, tmp_point=None, k=1):
    '''
    /更新近邻点和对应的最小距离集合
    min_dist_array为最小距离的集合
    NN为近邻点的集合
    tmp_dist和tmp_point分别是需要更新到min_dist_array，NN里的近邻点和距离
    '''
    
    if tmp_dist <= np.min(min_dist_array) : 
            for i in range(k-1,0,-1) :
                min_dist_array[i] = min_dist_array[i-1]
                NN[i] = NN[i-1]    
            min_dist_array[0] = tmp_dist
            NN[0] = tmp_point                
            return NN,min_dist_array
    for i in range(k) :
        if (min_dist_array[i] <= tmp_dist) and (min_dist_array[i+1] >= tmp_dist) :
            #tmp_dist在min_dist_array的第i位和第i+1位之间，则插入到i和i+1之间，并把最后一位给剔除掉
            for j in range(k-1,i,-1) : #range反向取值
                min_dist_array[j] = min_dist_array[j-1]
                NN[j] = NN[j-1]
            min_dist_array[i+1] = tmp_dist
            NN[i+1] = tmp_point
            break
    return NN,min_dist_array

''' 来源：https://blog.csdn.net/v_JULY_v/article/details/8203674, 2.5.1
    =====searchKDTree kd树最近邻查找 伪代码=====
    算法：k-d树最邻近查找
    输入：Kd，    //k-d tree类型
         target  //查询数据点
    输出：nearest， //最邻近数据点
         dist      //最邻近数据点和查询点间的距离
     
    1. If Kd为NULL，则设dist为infinite并返回
    2. //进行二叉查找，生成搜索路径
       Kd_point = &Kd；                   //Kd-point中保存k-d tree根节点地址
       nearest = Kd_point -> Node-data；  //初始化最近邻点
     
       while（Kd_point）
       　　push（Kd_point）到search_path中； //search_path是一个堆栈结构，存储着搜索路径节点指针
     
          If Dist（nearest，target） > Dist（Kd_point -> Node-data，target）
       　　　　nearest  = Kd_point -> Node-data；    //更新最近邻点
       　　　　Min_dist = Dist(Kd_point，target）；  //更新最近邻点与查询点间的距离  ***/
       　　s = Kd_point -> split；                       //确定待分割的方向
     
       　　If target[s] <= Kd_point -> Node-data[s]     //进行二叉查找
       　　　　Kd_point = Kd_point -> left；
       　　else
       　　　　Kd_point = Kd_point ->right；
       End while
     
    3. //回溯查找
       while（search_path != NULL）
       　　back_point = 从search_path取出一个节点指针；   //从search_path堆栈弹栈
       　　s = back_point -> split；                      //确定分割方向
     
       　　If Dist（target[s]，back_point -> Node-data[s]） < Max_dist   //判断还需进入的子空间
       　　　　If target[s] <= back_point -> Node-data[s]
       　　　　　　Kd_point = back_point -> right；  //如果target位于左子空间，就应进入右子空间
       　　　　else
       　　　　　　Kd_point = back_point -> left;    //如果target位于右子空间，就应进入左子空间
       　　　　将Kd_point压入search_path堆栈；
     
       　　If Dist（nearest，target） > Dist（Kd_Point -> Node-data，target）
       　　　　nearest  = Kd_point -> Node-data；                 //更新最近邻点
       　　　　Min_dist = Dist（Kd_point -> Node-data,target）；  //更新最近邻点与查询点间的距离的
       End while 
'''
def searchKDTree(KDTree=None, target_point=None, k=1):  
    '''
    /搜索kd树
    /输入值:KDTree,kd树;target_point,目标点；k,距离目标点最近的k个点的k值
    /输出值:k_arrayList,距离目标点最近的k个点的集合数组
    '''
    if(k == 0):
        return None
    #从根节点出发，递归地向下访问kd树。若目标点当前维的坐标小于切分点的坐标，则移动到左子节点，否则移动到右子节点
    tempNode = KDTree.root#定义临时节点，先从根节点出发
    NN = [tempNode.point] * k#定义最邻近点集合,k个元素，按照距离远近，由近到远。初始化为k个根节点
    min_dist_array = [float("inf")] * k#定义近邻点与目标点距离的集合.初始化为无穷大
    #     for i in range(k) :
    #         NN[i] = tempNode.point#定义最邻近点集合,k个元素，按照距离远近，由近到远。初始化为k个根节点以下往左的集合
    #         min_dist_array[i] = computeDist(NN[i],target_point)#定义近邻点与目标点距离的集合
    #         tempNode = tempNode.left
    nodeList = []#我们是用二分查找建立路径，定义依次查找节点的list

    def buildSearchPath(tempNode=None, nodeList=None, min_dist_array=None, NN=None, target_point=None):
        '''
        P:此方法是用来建立以tempNode为根节点，以下所有节点的查找路径，并将它们存放到nodeList中
        nodeList为一系列节点的顺序组合，按此先后顺序搜索最邻近点
        tempNode为"根节点",即以它为根节点，查找它以下所有的节点（空间）
        '''
        while tempNode :
            nodeList.append(tempNode)
            split = tempNode.split#节点的分割纬度
            point = tempNode.point#节点包含的数据,当前实例点
            tmp_dist = computeDist(point,target_point)
            if tmp_dist < np.max(min_dist_array) : #小于min_dist_array中最大的距离
                NN,min_dist_array = updateNN(min_dist_array, tmp_dist, NN, point, k)#更新最小距离和最邻近点
            if  target_point[split] <= point[split] : #如果目标点当前维的值小于等于切分点的当前维坐标值，移动到左节点
                tempNode = tempNode.left
            else : #如果目标点当前维的值大于切分点的当前维坐标值，移动到右节点
                tempNode = tempNode.right
        return NN,min_dist_array
    #建立查找路径
    NN,min_dist_array = buildSearchPath(tempNode,nodeList,min_dist_array, NN, target_point)
    #回溯查找
    while nodeList :
        back_node = nodeList.pop()#将nodeList里的元素从后往前一个个推出来
        split = back_node.split
        point = back_node.point
        #判断是否需要进入父节点搜素
        #如果当前纬度，目标点减实例点大于最小距离，就没必要进入父节点搜素了
        #因为目标点到切割超平面的距离很大，那邻近点肯定不在那个切割的空间里，即没必要进入那个空间搜素了
        if not abs(target_point[split] - point[split]) >= np.max(min_dist_array) :
            #判断是搜索左子节点，还是搜索右子节点
            if (target_point[split] <= point[split]) :
                #如果目标点在左子节点的空间，则搜索右子节点，查看右节点是否有更邻近点
                tempNode = back_node.right
            else :
                #如果目标点在右子节点的空间，则搜索左子节点，查看左节点是否有更邻近点
                tempNode = back_node.left
            
            if tempNode :
                #把tempNode（此时它为另一个全新的未搜素的空间，需要将它放入nodeList，进行最近邻搜索）放入nodeList
                #nodeList.append(tempNode)
                #不能单纯地将tempNode存放到nodeList，这样下次只会搜索这一个节点
                #因为tempNode可做为一个全新的空间，故而需重新以它为根节点，构建查找路径，搜索它名下所有的节点
                NN,min_dist_array = buildSearchPath(tempNode,nodeList,min_dist_array, NN, target_point)
    #                 curr_dist = computeDist(tempNode.point,target_point)
                    #是否该节点为更邻近点，如果是，赋值给最邻近点
    #                 if curr_dist < np.max(min_dist_array) :
    #                     NN,min_dist_array = updateNN(min_dist_array, curr_dist, NN, tempNode.point, k)#更新最小距离和最邻近点
    return NN,min_dist_array 

def KNN_kdTree(train_X_list, train_Y_list, test_X, k):
    '''
    k近邻算法的分类器
    输入：
    test_X:目标点
    train_X_list:训练点集合
    train_Y_list:训练点对应的标签
    k:k值
    这个方法的目的：已知训练点train_X_list和对应的标签labels，确定目标点test_X对应的labels
    ''' 
    kd = kdTree(train_X_list)#构建train_X_list的kd树
    NN,min_dist_array = searchKDTree(kd, test_X, k)#搜索kd树，返回最近的k个点的集合NN，和对应的距离min_dist_array
    voteIlabels = []
    #多数投票法则确定inX的标签，为防止边界处分类不准的情况，以距离的倒数为权重，即距离越近，权重越大，越该认为inX是属于该类
    for i in range(k) :
        #找到每个近邻点对应的标签
        nni = list(NN[i])
        voteIlabels.append(train_Y_list[train_X_list.index(nni)])
        
    #     #开始记数,加权重的方法
    #     uniques = np.unique(voteIlabels)
    #     counts = [0.0] * len(uniques)
    #     for i in range(len(voteIlabels)) :
    #         for j in range(len(uniques)) :
    #             if voteIlabels[i] == uniques[j] :
    #                 counts[j] = counts[j] + uniques[j] / min_dist_array[i] #权重为距离的倒数
    #                 break
    #开始记数,不加权重的方法
    uniques, counts = np.unique(voteIlabels, return_counts=True)
    return uniques[np.argmax(counts)]


if __name__ == '__main__':
    dataAndDraw_fun = dataAndDraw()

    #0- 数据集准备
    [X_list, Y_list] = dataAndDraw_fun.produceData()
    drawPoints_X = [ X[0] for X in X_list ]
    drawPoints_Y = [ X[1] for X in X_list ]

    #1- KNN_simple 测试
    test_X = [6, 2]
    test_X_buffer = test_X[:]
    start_simple = time.clock()
    KNN_simple_test = KNN_simple(X_list, Y_list, test_X, 5, 2)
    predictLabel_simple = KNN_simple_test.predict()
    end_simple = time.clock()
    print('KNN_simple- test_X Label is:', predictLabel_simple)
    print('\t-KNN_simple 用时：', (end_simple-start_simple), '秒')
    ## 画图
    dataAndDraw_fun.drawDots(drawPoints_X[:50], drawPoints_Y[:50], '*', 'red', '-1')
    dataAndDraw_fun.drawDots(drawPoints_X[50:100], drawPoints_Y[50:100], 'o', 'blue', '+1')
    dataAndDraw_fun.drawDots(test_X_buffer[0], test_X_buffer[1], 'x', 'green', 'test_X')
    plt.title('KNN_simple(original)')
    plt.show()

    #2- KNN_kdTree 测试
    start_kdTree = time.clock()
    predictLabel_kdTree = KNN_kdTree(X_list, Y_list, test_X, 5)
    end_kdTree = time.clock()
    print('KNN_kdTree- test_X Label is:', predictLabel_kdTree)
    print('\t-KNN_kdTree 用时：', (end_kdTree - start_kdTree), '秒')