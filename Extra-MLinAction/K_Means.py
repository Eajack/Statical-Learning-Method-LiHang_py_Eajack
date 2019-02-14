'''
!/usr/bin/env python3
-*- coding : utf-8 -*-
Author: Eajack
date:2019/2/11 - 2019/2/13
Function：
	额外：
	《机器学习实战-第10章》
		1- K-Means均值聚类
	部分参考：https://www.jianshu.com/p/5314834f9f8e
	
'''

import random, copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def loadData():
	iris = datasets.load_iris()
	X, y = iris.data, iris.target
	data = X[:,[1,2]] # 为了便于可视化，只取两个维度
	return np.mat(data,dtype=np.float)

def KMeans(k, train_data_Mat):
	def _calulateDistance(p1_array, p2_array):
		# 欧式距离
		tmp = np.sum(np.power((p1_array-p2_array),2))
		return np.sqrt(tmp)

	def _finished(label_array1, label_array2):
		label_list1 = label_array1.tolist()
		label_list2 = label_array2.tolist()

		return (label_list1 == label_list2)

	def _produce_randK_points(dataSet, k):
		dim = dataSet.shape[1]
		center_mat = np.mat(np.zeros((k,dim)))
		for j in range(dim):
			minJ = min(dataSet[:,j])
			maxJ = max(dataSet[:,j])
			rangeJ = float(maxJ-minJ)
			center_mat[:,j] = minJ + rangeJ * np.random.rand(k,1)
		return center_mat

	#1- 初始化k个随机聚类中心
	dataNum, dim = np.shape(train_data_Mat)
	label = np.zeros(dataNum, dtype=np.int)
	center_mat = _produce_randK_points(train_data_Mat,k)

	#2- 开始循环
	finishFlag = False
	while(not finishFlag):
		old_label = copy.deepcopy(label)
		#3- 遍历所有数据点，按距离最小分配簇
		for i in range(dataNum):
			minDist, minIndex = np.inf, -1
			for j in range(k):
				dist = _calulateDistance(train_data_Mat[i,:], center_mat[j,:])
				if(dist < minDist):
					minDist, minIndex = dist, j
			label[i] = minIndex

		#4- 更新center
		for i in range(k):
			center_i_add = np.zeros(np.shape(center_mat[0]))
			i_count = 0
			for label_index in range(dataNum):
				if(label[label_index] == i):
					i_count += 1
					center_i_add += train_data_Mat[label_index]
			center_mat[i] = center_i_add / i_count

		#5- 判断是否完成
		finishFlag = _finished(label, old_label)

	return center_mat, label

def draw(train_data_Mat, center_mat, label_array):
	#1- 画原数据图
	for index in range(len(label_array)):
		dot = train_data_Mat[index].tolist()
		if(label_array[index] == 1):
			plt.scatter(dot[0][0],dot[0][1],c='r')
		else:
			plt.scatter(dot[0][0],dot[0][1],c='b')

	#2- 画聚类中心
	for center in center_mat:
		center = center.tolist()[0]
		plt.scatter(center[0],center[1],marker='*',s=300,c='g')

	plt.show()


if __name__ == '__main__':
	train_data_Mat = loadData()
	center_mat, label = KMeans(2, train_data_Mat)
	draw(train_data_Mat, center_mat, label)