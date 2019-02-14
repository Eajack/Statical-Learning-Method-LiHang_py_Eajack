'''
!/usr/bin/env python3
-*- coding : utf-8 -*-
Author: Eajack
date:2019/1/20 - 2019/1/22
Function：
	《统计学习方法-李航》 第六章
	难点：书中的向量（矩阵）不知是点积 or 矩阵相乘，搞得很晕
	1- Logistic回归
						  N
		似然函数 L(w) = SIGMA( x_i(y_i-(1/(1+exp(-w*x)))) )
						 i=1
		求L(w)极大值，此时
			“L(w)用梯度上升法” <=> “-L(w)用梯度下降法”
			都是 w = w + ▽f(w)
	1.1 GD梯度下降算法
		-BGD
		-SGD
		-MBGD
	1.2 牛顿法（弄太久了，不搞了。。。）
	2- 最大熵模型（弄太久了，不搞了）

	参考(1)- 《机器学习实战-第5章》，该书思路和《统计学习方法》的不同
		(2)- https://zhuanlan.zhihu.com/p/28057866
		(3)- https://zhuanlan.zhihu.com/p/51024390
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import copy, warnings, time 
warnings.filterwarnings(action = 'ignore', category = RuntimeWarning)

np.set_printoptions(precision=10)

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
	Y_list = list(np.zeros(50))
	Y_list.extend(list(np.ones(50)))
	#print(X_list, Y_list)
	return [X_list, Y_list]

class LR(object):
	"""docstring for LR"""
	def __init__(self, maxloop, epsilon, step, miniBatch_MBGD=10):
		self.maxloop = maxloop
		self.epsilon = epsilon
		self.step = step
		self.miniBatch_MBGD = 10
		self.w_min = 100000000

	#(1)- 通用函数包
	# 梯度上升法
	def sigmoid(self, X):
		return 1.0/(1+np.exp(-X))

	def costFunction(self, w, train_X_mat, train_Y_mat):
		m = train_X_mat.shape[0]
		h = self.sigmoid(train_X_mat.dot(w))
		J = (-1.0/m)*(np.log(h).T.dot(train_Y_mat) + np.log(1 - h).T.dot(1-train_Y_mat))
		return J

###
	# # 牛顿法
	# def likehoodFunction_Grad1(self, w, train_X_mat, train_Y_mat):
	# 	# grad1 = X_T * (Y-P)，点积（按元素相乘，求和）
	# 	p_mat = np.ones((train_X_mat.shape[0],1))

	# 	for i in range(train_X_mat.shape[0]):
	# 		p_buffer1 = np.multiply(train_X_mat[i,:],w)
	# 		p_buffer2 = sum(p_buffer1.tolist()[0])
	# 		p_mat[i][0] = -1/( 1+np.exp(p_buffer2) )

	# 	grad_mat = np.multiply(train_X_mat,train_Y_mat-p_mat)
	# 	grad1 = np.zeros((1,grad_mat.shape[1]))
	# 	for i in range(grad_mat.shape[0]):
	# 		grad1 += grad_mat[i,:]

	# 	return grad1

	# def likehoodFunction_Grad2(self, w, train_X_mat, train_Y_mat):
	# 	# grad2 = X_T * Z * X，点积（按元素相乘，求和）
	# 	p_mat = np.ones((train_X_mat.shape[0],1))
	# 	Z_mat = np.ones(np.shape(p_mat))

	# 	for i in range(train_X_mat.shape[0]):
	# 		p_buffer1 = np.multiply(train_X_mat[i,:],w)
	# 		p_buffer2 = sum(p_buffer1.tolist()[0])
	# 		p = -1/( 1+np.exp(p_buffer2) )
	# 		p_mat[i][0] = p
	# 		Z_mat[i][0] = p * (1 - p)

	# 	grad_mat = np.multiply( (np.power(train_X_mat,2)), Z_mat )
	# 	grad2 = np.zeros((1,grad_mat.shape[1]))
	# 	for i in range(grad_mat.shape[0]):
	# 		grad2 += grad_mat[i,:]

	# 	return grad2
###	

	def draw(self, train_X_list, train_Y_list, w):
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
			minX += 0.01
			y = ((-w[0]*minX-w[2])/w[1]).tolist()
			plt.scatter(minX, y, c='r', marker='.')

		plt.title('Results')
		plt.show()

	#(2)- 批量梯度下降Batch Gradient Descent(上升)
	# 结果显示，可以达到全局最优解，cost收敛于0
	def train_BGD(self, train_X_list, train_Y_list):
		train_X_list_clone = copy.deepcopy(train_X_list)
		for X in train_X_list_clone:
			X.append(1)

		train_X_mat = np.mat(train_X_list_clone)
		train_Y_mat = np.mat(train_Y_list).transpose()
		m,n = np.shape(train_X_mat)
		w = np.mat(np.ones((n,1)))
		iter_list = []
		cost_list = []
		#GD算法关键代码
		##########################
		for k in range(self.maxloop):
			value = self.sigmoid(train_X_mat * w)
			error = (train_Y_mat - value)
			cost = self.costFunction(w, train_X_mat, train_Y_mat)
			w = w + (1.0/m) * self.step * train_X_mat.transpose() * error #(上升)
			iter_list.append(k)
			cost_list.append(cost)
		##########################

		plt.figure()
		plt.title('BGD Cost-Iter Function')
		plt.scatter(iter_list, cost_list, c='r', marker='.')
		plt.show()
		
		self.w_min = w
		self.draw(train_X_list, train_Y_list, w)

	#(3)- 随机梯度下降Stochastic Gradient Descent(上升)
	# 结果显示，不一定能达到全局最优解，cost不一定收敛于0
	def train_SGD(self, train_X_list, train_Y_list):
		train_X_list_clone = copy.deepcopy(train_X_list)
		for X in train_X_list_clone:
			X.append(1)

		train_X_mat = np.mat(train_X_list_clone)
		train_Y_mat = np.mat(train_Y_list).transpose()
		m,n = np.shape(train_X_mat)
		w = np.mat(np.ones((n,1)))
		iter_list = []
		cost_list = []
		for i in range(m):
			value = self.sigmoid(train_X_mat[i] * w)
			error = (train_Y_mat[i] - value)
			cost = self.costFunction(w, train_X_mat, train_Y_mat)
			w = w + (1.0/m) * self.step * train_X_mat[i].transpose() * error #(上升)
			iter_list.append(i)
			cost_list.append(cost)

		plt.figure()
		plt.scatter(iter_list, cost_list, c='r', marker='.')
		plt.title('SGD Cost-Iter Function')
		plt.show()

		self.w_min = w
		self.draw(train_X_list, train_Y_list, w)

	#(4)- 小批量梯度下降法 Mini-Batch Gradient Descent(上升)
	# 结果显示，不一定能达到全局最优解，cost不一定收敛于0
	def train_MBGD(self, train_X_list, train_Y_list):
		train_X_list_clone = copy.deepcopy(train_X_list)
		for X in train_X_list_clone:
			X.append(1)

		train_X_mat = np.mat(train_X_list_clone)
		train_Y_mat = np.mat(train_Y_list).transpose()
		m,n = np.shape(train_X_mat)
		w = np.mat(np.ones((n,1)))
		iter_list = []
		cost_list = []
		for i in range(m-self.miniBatch_MBGD):
			value = self.sigmoid(train_X_mat[i:self.miniBatch_MBGD] * w)
			error = (train_Y_mat[i:self.miniBatch_MBGD] - value)
			cost = self.costFunction(w, train_X_mat, train_Y_mat)
			w = w + (1.0/self.miniBatch_MBGD) * self.step * train_X_mat[i:self.miniBatch_MBGD].transpose() * error #(上升)
			iter_list.append(i)
			cost_list.append(cost)

		plt.figure()
		plt.scatter(iter_list, cost_list, c='r', marker='.')
		plt.title('MBGD Cost-Iter Function')
		plt.show()

		self.w_min = w
		self.draw(train_X_list, train_Y_list, w)

###
	# #(5)- 拟牛顿法
	# def train_NT(self, train_X_list, train_Y_list):
	# 	#(1)- 准备
	# 	train_X_list_clone = []
	# 	for X in train_X_list:
	# 		X_buffer = copy.deepcopy(X)
	# 		X_buffer.append(1)
	# 		train_X_list_clone.append(X_buffer)
	# 	train_X_mat = np.mat(train_X_list_clone)

	# 	train_Y_list_clone = []
	# 	for Y in train_Y_list:
	# 		train_Y_list_clone.append([Y])
	# 	train_Y_mat = np.mat(train_Y_list_clone)
	# 	self.w_min = np.mat(np.zeros( (1, len(train_X_list[0])+1) ))

	# 	#(2)- 循环
	# 	w_buffer = [843.9,447.9,150]
	# 	self.draw(train_X_list, train_Y_list, w_buffer)
	# 	grad1 = 1000000
	# 	while( np.linalg.norm(grad1) >= self.epsilon ):
	# 		w = self.w_min
	# 		grad1 = self.likehoodFunction_Grad1(w, train_X_mat, train_Y_mat)
	# 		grad2 = self.likehoodFunction_Grad2(w, train_X_mat, train_Y_mat)
	# 		print(grad1)
	# 		self.w_min = w + (grad1/grad2)


	# 	return self.w_min
###
	#(6)- 测试函数
	def test(self, text_X):
		if(self.w_min == 100000000):
			print("Error: 还没训练，别测试")
			return 0
		text_X_clone = copy.deepcopy(text_X)
		text_X_clone.append(1)

		P_1 = np.exp(np.vdot(self.w_min, text_X_clone)) / (1 + np.exp(np.vdot(self.w_min, text_X_clone)))
		if(P_1 >= 0.5):
			print('测试样本类别：1（正类）')
			return 1
		else:
			print('测试样本类别：-1（负类）')
			return -1


if __name__ == '__main__':
	[train_X_list, train_Y_list] = produceData()

	#1- LR回归
	LR_Node = LR(10000, 0.001, 0.001)
	#1.1- GD梯度下降法
	start1 = time.clock()
	LR_Node.train_BGD(train_X_list, train_Y_list)
	end1 = time.clock()
	print('1- BGD耗时：{}s'.format(end1-start1))

	start2 = time.clock()
	LR_Node.train_SGD(train_X_list, train_Y_list)
	end2 = time.clock()
	print('2- SGD耗时：{}s'.format(end2-start2))

	start3 = time.clock()
	LR_Node.train_MBGD(train_X_list, train_Y_list)
	end3 = time.clock()
	print('3- MBGD耗时：{}s'.format(end3-start3))

	# #1.2- 拟牛顿法
	# start4 = time.clock()
	# LR_Node.train_NT(train_X_list, train_Y_list)
	# end4 = time.clock()
	# print('4- 拟牛顿法耗时：{}s'.format(end4-start4))
	