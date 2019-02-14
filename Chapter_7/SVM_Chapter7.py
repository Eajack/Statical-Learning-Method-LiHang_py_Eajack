'''
!/usr/bin/env python3
-*- coding : utf-8 -*-
Author: Eajack
date:2019/1/22 - 2019/1/24
Function：
	《统计学习方法-李航》 第七章
	1- LSVML（线性可分支持向量机）（实现）
	2- LSVM（线性SVM）& NLSVM（非线性SVM）不实现了
	3- PS：LSVM & LSVML区别：前者有个惩罚系数C
		   NLSVM & LSVML区别：前者有个惩罚系数C、有核函数
	
'''
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import simpleSMO, OthersSVM

def produceData():
	# 借助于load_iris
	# 数据集较小，100个
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
	plt.title('Original Data')
	plt.show()
	#plt.hold()

	X_list = [ [df[0:100]['X'][i],df[0:100]['Y'][i]] for i in range(0,100) ]
	Y_list = list(-np.ones(50))
	Y_list.extend(list(np.ones(50)))
	#print(X_list, Y_list)
	return X_list, Y_list

def SVM_plot(train_X_list, train_Y_list, w, b, alphas):
	#(1)- 画原散点图
	fig = plt.figure()
	plt.scatter([X[0] for X in train_X_list[:50]], [X[1] for X in train_X_list[:50]], label='-1')
	plt.scatter([X[0] for X in train_X_list[50:100]], [X[1] for X in train_X_list[50:100]], label='1')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.legend()

	#(2)- 画分割线
	minX = min([X[0] for X in train_X_list])
	maxX = max([X[0] for X in train_X_list])
	while(minX < maxX):
		minX += 0.001
		y = ((-w[0]*minX-b)/w[1]).tolist()
		plt.scatter(minX, y, c='r', marker='.')

	plt.title('LSVML-Results')

	#(3)- 画支持变量
	for i, X in enumerate(train_X_list):
		if(alphas[i] > 0.01):
			print(X, train_Y_list[i], alphas[i])
			plt.scatter(X[0], X[1], color='', marker='o', edgecolors='g', s=200)

	plt.show()


class SVM():
	"""docstring for SVM"""
	def __init__(self):
		pass

	def train_LSVML(self, train_X_list, train_Y_list, C, toler, maxLoopTime, kernelOption):
		'''
			Linear support vector machine in linearly separable case
				线性可分支持向量机，对偶问题求解
			输入数据：线性可分
		'''
		train_X_mat = np.mat(train_X_list)
		train_Y_mat = np.mat(train_Y_list)
		#1- 简化版SMO求解
		#alphas, b = OthersSVM.trainSVM(train_X_mat, train_Y_mat.T, C, toler, maxLoopTime, kernelOption)
		alphas, b =  simpleSMO.SMO_simple(train_X_list, train_Y_list, C, toler, maxLoopTime, kernelOption)
		
		#2- 通过alphas求w, b
		w = np.zeros( (len(train_X_list[0]),1) )
		alphas_mat = np.mat(alphas)
		w_coe = np.multiply(alphas_mat,train_Y_mat.T)
		w_coe = w_coe.tolist()
		for i, x in enumerate(train_X_mat):
			w += w_coe[i][0] * x.T
		print('w=====================W',w)
		# b = 0
		# for j, a in enumerate(alphas):
		# 	if(a > 0):
		# 		b_buffer = 0
		# 		for i, x in enumerate(train_X_mat):
		# 			b_buffer += w_coe[i][0] * (x*train_X_mat[j].T)
		# 		b_buffer = b_buffer.tolist()
		# 		b_buffer = b_buffer[0]
		# 		b = train_Y_list[j] - b_buffer
		# 		break

		#3- 画超平面，仅限于2维变量
		SVM_plot(train_X_list, train_Y_list, w, b, alphas)
		


if __name__ == '__main__':
	train_X_list, train_Y_list = produceData()
	SVM_node = SVM()
	
	#1- 线性可分支持向量机
	SVM_node.train_LSVML(train_X_list, train_Y_list, 0.6, 0.001, 100, ('linear',0))