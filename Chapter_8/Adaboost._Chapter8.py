'''
!/usr/bin/env python3
-*- coding : utf-8 -*-
Author: Eajack
date:2019/1/29 - 2019/1/30
Function：
	《统计学习方法-李航》 第八章
	1- Adaboost算法，例8.1
	2- 提升树（不实现，只看书，用了梯度上升法）
		提升树 <=> Adaboost中的基分类器是树（一般是决策树），
					损失函数为平方误差函数，
	
'''
import numpy as np
import copy
import matplotlib.pyplot as plt

def produceData0():
	'''
		书中P140例8.1数据
	'''
	train_X_list = list(range(10))
	train_Y_list = [1,1,1,-1,-1,-1,1,1,1,-1]

	return train_X_list, train_Y_list

def Adaboost0(train_X_list, train_Y_list, M=3):
	'''
		书中P140例8.1，弱分类器：x<v & x>v
		M为弱分类器个数
		结果：和书本有点出入了
	'''
	#1- 初始化权值
	train_num = len(train_Y_list)
	D = [1.0/train_num] * train_num
	v_list = [ (train_X_list[index]+train_X_list[index+1])/2 for index in range(train_num-1)]
	adaboost_classifer_list = []

	#2- 循环开始
	for m in range(1,M+1):
		#2.1- 获得误差率最低的v
		# 如果和之前重复，则选第二低的v，以此类推
		e_list = []
		for v in v_list:
			underV_list = train_X_list[0:int(v)+1]
			aboveV_list = train_X_list[int(v)+1:]
			underV_y_list = [1] * len(underV_list)
			aboveV_y_list = [-1] * len(aboveV_list)
			predict_y_list = copy.deepcopy(underV_y_list)
			predict_y_list.extend(aboveV_y_list)
			#2.2- 计算e
			e = 0
			for index in range(train_num):
				if(predict_y_list[index] != train_Y_list[index]):
					e += D[index] * 1

			e_list.append(round(e,4))

		# 获取最佳v值
		best_e = 0
		best_v = 0
		if(adaboost_classifer_list):
			past_v_list = [ item[2] for item in adaboost_classifer_list]

			while(1):
				best_index = e_list.index(min(e_list))
				best_e = e_list[best_index]
				best_v = v_list[best_index]

				if(best_v not in past_v_list):
					break
				else:
					del e_list[best_index]
					del v_list[best_index]
		else:
			past_v_list = []
			best_e = min(e_list)
			best_v = v_list[e_list.index(best_e)]


		#2.3- 计算G(x)系数
		alpha = round( (1/2) * np.log((1-best_e)/best_e), 4 )

		#2.4- 更新权值D
		# 获得G(x)输出y
		underV_list = train_X_list[0:int(best_v)+1]
		aboveV_list = train_X_list[int(best_v)+1:]
		underV_y_list = [1] * len(underV_list)
		aboveV_y_list = [-1] * len(aboveV_list)
		predict_y_list = copy.deepcopy(underV_y_list)
		predict_y_list.extend(aboveV_y_list)

		# 计算Z
		Z = [ D[index] * np.exp((-alpha)*train_Y_list[index]*predict_y_list[index])\
			 for index in range(train_num)]
		Z = sum(Z)

		for index in range(train_num):
			if(predict_y_list[index] == train_Y_list[index]):
				D[index] = round( (D[index]/Z) * np.exp(-alpha), 4 )
			else:
				D[index] = round( (D[index]/Z) * np.exp(alpha), 4 )

		D_clone = copy.deepcopy(D)
		adaboost_classifer_list.append( (alpha, best_v, D_clone, best_e) )

	return adaboost_classifer_list

def draw0(train_X_list, train_Y_list, adaboost_classifer_list):
	#1- 画原散点图
	plt.figure()
	for index in range(len(train_Y_list)):
		if(train_Y_list[index] == 1):
			plt.scatter(train_X_list[index], 1, \
				color='w', marker='.', s=200, label='1')
		else:
			plt.scatter(train_X_list[index], -1, \
				color='k', marker='.', s=200, label='-1')
	#2- 画分类器
	alpha_list = []
	v_list = []
	for item in adaboost_classifer_list:
		alpha_list.append(item[0])
		v_list.append(item[1])

	x_draw_list = np.arange(0, 10, 0.01)
	x_draw_list.tolist()
	x_draw_list = [ round(x_draw, 4) for x_draw in x_draw_list ]
	y_draw_list = []
	for x_draw in x_draw_list:
		y_buffer = 0
		for v_index in range(len(v_list)):
			if(x_draw < v_list[v_index]):
				y_buffer += alpha_list[0]
			else:
				y_buffer += -alpha_list[0]
		if(y_buffer > 0):
			y_draw_list.append(1)
			plt.scatter(x_draw, 1, \
				color='r', marker='o', s=50, label='1')
		elif(y_buffer < 0):
			y_draw_list.append(-1)
			plt.scatter(x_draw, -1, \
				color='g', marker='o', s=50, label='-1')
		else:
			y_draw_list.append(0)


	plt.show()

if __name__ == '__main__':
	train_X_list, train_Y_list = produceData0()

	#1- Adaboost 例题8.1
	adaboost_classifer_list = Adaboost0(train_X_list, train_Y_list)
	draw0(train_X_list, train_Y_list, adaboost_classifer_list)
	