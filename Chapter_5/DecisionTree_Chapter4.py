'''
!/usr/bin/env python3
-*- coding : utf-8 -*-
Author: Eajack
date:2018/12/13 - 2018/12/22
Function：
	《统计学习方法-李航》 第五章（1-5章，耗时最长1章……）
	DT 三要素：特征选择、树的生成、树的剪枝
	1- ID3 & C4.5
	   1)- 生成（标称值，即属性不是数值连续型）
	   2)- 分类
	   3)- 剪枝(后剪枝, Failed)（不是P66的算法5.4，而是选用了《机器学习-周志华》的P82的后剪枝）
	   	（感觉剪枝实在太烦了，或者思路有问题……没意志写下去了……通用函数集中isLeafNode、getLeafNodeNum、getleafNodesPathsFile、
	   		getleafNodesPaths都是为后剪枝做准备，光写剪枝，就花了3天有空时间……so, 放弃了）
	   4)- 通用函数集：获取所有根节点 => 叶节点路径、获取DT叶节点数
	2- CART(属于二叉树)
		1)- 生成：LSRT & CART树，二者区别在于，前者输入连续（回归），后者输入离散（分类），思路一致
		2)- 分类：由于决策树分类思路和1的一致，不重复写了
		3)- 剪枝：算法5.7，自底向上，对有2个叶节点的内部节点剪枝，后剪枝策略，不包括Step(7)的交叉验证
		(Attention: 不知为啥测试数据得到的CART，多次运行有可能生成不一样。。。不知是不是代码写错了，但也不管了…)

'''
import numpy as np
from collections import Counter
import copy, math, time, os, json
import plotDT



class ID3orC45_DecisionTree(object):
	"""
		决策树形式：字典形式，例如 DT = {"有工作": {"有": "是","没有": "否"}
	"""
	def __init__(self, epsilon=0.1):
		self.epsilon = epsilon	# 阈值epsilon
		self.classResult = 0

		pathsFile = open('leafNodePaths.txt', 'a', encoding='utf-8')
		self.pathsFile = pathsFile
		
	#0- 通用函数集
	def chooseBestFeature(self, train_X_list, train_X_labels, train_Y_list, DT_method):
		'''

		:param train_X_list: 训练集X
		:param train_Y_list: 训练集对应标签
		:param DT_method: 1 => ID3算法， 2 => C4.5算法
		:return: bestFeatureIndex, bestFeatureName
		'''
		bestValue = 0
		bestFeatureIndex = 0
		bestFeatureName = ''
		if(len(train_X_labels) != len(train_X_list[0])):
			print("chooseBestFeature 出错！label个数和train_X_list列数不一致！")
			return None
		IG_list = [ self.get_InformetionGain(train_X_list, train_Y_list, train_X_dim) \
								 for train_X_dim in range(len(train_X_labels)) ]
		IGR_list = [ self.getInformationGainRatio(train_X_list, train_Y_list, train_X_dim) \
								 for train_X_dim in range(len(train_X_labels)) ]
		if(DT_method == 1):
			bestValue = max(IG_list)
			bestFeatureIndex = IG_list.index(bestValue)
			bestFeatureName = train_X_labels[bestFeatureIndex]
		elif(DT_method == 2):
			bestValue = max(IGR_list)
			bestFeatureIndex = IGR_list.index(max(IGR_list))
			bestFeatureName = train_X_labels[bestFeatureIndex]

		return bestValue, bestFeatureIndex, bestFeatureName

	def split_trainSets(self, train_X_list, train_X_labels, train_Y_list, bestFeatureIndex):
		#(1)- train_X_list & train_Y_list
		delete_label_col_list = [ train_X[bestFeatureIndex] for train_X in train_X_list ]
		delete_label_col_valueKinds_list = list(set(delete_label_col_list))
		new_train_XMatrix_list = [0] * len(delete_label_col_valueKinds_list)
		new_train_YMatrix_list = [0] * len(delete_label_col_valueKinds_list)

		for index in range(len(delete_label_col_valueKinds_list)):
			# new_train_XMatrix_list
			now_deleteKind = delete_label_col_valueKinds_list[index]
			now_deleteKind_index_list = [ index1 for index1 in range(len(train_X_list)) \
										  if(train_X_list[index1][bestFeatureIndex] == now_deleteKind) ]

			# 注意！！！；此处要深拷贝train_X_list，因为浅拷贝（或直接用）会改变train_X_list
			new_train_XMatrix_buffer = [ copy.deepcopy(train_X_list)[index2] for index2 in now_deleteKind_index_list ]

			for NTXvalue_list in new_train_XMatrix_buffer:
				del(NTXvalue_list[bestFeatureIndex])
			new_train_XMatrix_list[index] = new_train_XMatrix_buffer

			# new_train_YMatrix_list
			new_train_YMatrix_buffer = [ train_Y_list[now_deleteKind_index]\
				for now_deleteKind_index in now_deleteKind_index_list ]
			new_train_YMatrix_list[index] = new_train_YMatrix_buffer

		#(2)- train_X_labels
		new_train_X_labels = [ train_X_labels[index] for index in range(len(train_X_labels)) \
			if(index != bestFeatureIndex) ]

		return new_train_XMatrix_list, new_train_X_labels, new_train_YMatrix_list

	def isLeafNode(self, DT):
		if( (type(DT).__name__) != 'dict' ):
			return True
		DT_values = []
		for value in DT.values():
			DT_values.append(value)
		if( len(DT_values) == 1 and type(DT_values[0]).__name__ != 'dict' ):
			return True
		else:
			return False

	def getLeafNodeNum(self, DT):
		DT_buffer = copy.deepcopy(DT)
		leafNodeNum = 0
		for value in DT_buffer.values():
			if(self.isLeafNode(value)):
				leafNodeNum += 1
			else:
				leafNodeNum += self.getLeafNodeNum(value)

		return leafNodeNum

	def getleafNodesPathsFile(self, DT):
		DT_buffer = copy.deepcopy(DT)
		if(type(DT).__name__ != 'dict'):
			print(DT,file=self.pathsFile)
			return

		for key, value in DT_buffer.items():
			value_list = [ i for i in value ]
			for value in value_list:
				print((key, ':', value), end=' => ', file=self.pathsFile)
				if( len(value_list) == 1 and type(value_list[0]).__name__ != 'dict' ):
					print('\n', file=self.pathsFile)
					return
				else:
					self.getleafNodesPathsFile(DT_buffer[key][value])

	def getleafNodesPaths(self, DT, DT_root):
		self.getleafNodesPathsFile(DT)
		self.pathsFile.close()
		leafNodePaths_list = []
		with open('leafNodePaths.txt', 'r', encoding='utf-8') as pathFile:
			oldPath_list = pathFile.readlines()
			for index, oldPath in enumerate(oldPath_list):
				oldPath_new = oldPath.strip().replace("'", '').replace(' ','').\
					replace('(','').replace(')', '').replace('=>',',').\
					replace(',:,',',')
				oldPath_list[index] = oldPath_new
			#print('=========================')

			for index, oldPath_now in enumerate(oldPath_list):
				if(DT_root not in oldPath_now):
					oldPath_before = oldPath_list[index-1]

					oldPath_now_firstWord = oldPath_now[0:oldPath_now.index(',')]
					tragetIndex = oldPath_before.find(oldPath_now_firstWord)

					newPath_now = oldPath_before[0:tragetIndex] + (oldPath_now)
					oldPath_list[index] = newPath_now
					leafNodePaths_list.append(newPath_now)
				else:
					leafNodePaths_list.append(oldPath_now)

		leafNodePaths_list.sort(key = lambda i:len(i))
		#[ print(path) for path in leafNodePaths_list ]
		os.remove('leafNodePaths.txt')
		self.pathsFile = open('leafNodePaths.txt', 'a', encoding='utf-8')
		return leafNodePaths_list


	#1- ID3 & C4.5 DT算法
	# Attention: 二者区别仅在于：信息增益 & 信息增益率 作为特征划分标准
	#1.1- 获得信息增益/信息增益率
	def get_InformetionGain(self, train_X_list, train_Y_list, train_X_dim):
		'''
		:param: train_X_list
		:param: train_Y_list
		:param: train_X_dim：训练集点X的某一个维度索引数值，如 0，1...
			eg: train_X_list = [ [1,2], [2,3] ] = [ train_X_1, train_X_2 ]
				train_X_2(train_X_dim) 指的是train_X_1[1], train_X_2[1]对应维度 X_2
		:return: 计算g(train_X_list, train_X_dim)，即信息增益
			g(D|A) = H(D) - H(D|A)
		'''

		#(1)- 计算H(D)
		trainNum = len(train_Y_list)
		train_Y_kinds_list = list(set(train_Y_list))
		H_D_P = [ (train_Y_list.count(train_Y)/trainNum) for train_Y in train_Y_kinds_list ]
		H_D = [ (p*math.log(1/p,2)) for p in H_D_P ]
		H_D = sum(H_D)

		#(2)- 计算H(D|A)
		#(2.1)- 计算A对应的种类概率list
		train_X_dim_list = [ train_X[train_X_dim] for train_X in train_X_list ]
		train_X_dim_kinds_list = list(set(train_X_dim_list))
		train_X_dim_p_list = []
		for train_X_dim_item in train_X_dim_kinds_list:
			train_X_dim_p_list.append(train_X_dim_list.count(train_X_dim_item)/trainNum)
		#(2.2)- 计算按照A对train_Y_list划分后，每部分熵值
		train_DA_dim_p_list = []
		for entropy_count in range(len(train_X_dim_p_list)):
			train_Y_divided_list = []
			train_X_item_divided = train_X_dim_kinds_list[entropy_count]
			for train_Y_count in range(len(train_Y_list)):
				if(train_X_dim_list[train_Y_count] == train_X_item_divided):
					train_Y_divided_list.append(train_Y_list[train_Y_count])
			train_DA_Num = len(train_Y_divided_list)
			train_DA_Y_kinds_list = list(set(train_Y_divided_list))
			H_DA_P = [(train_Y_divided_list.count(train_Y) / train_DA_Num) for train_Y in train_DA_Y_kinds_list]
			H_DA_divided = [(p * math.log(1/p,2)) for p in H_DA_P]
			H_DA_divided = sum(H_DA_divided)
			train_DA_dim_p_list.append(H_DA_divided)
		#(2.3)- 计算H(D|A)
		H_DA = np.vdot(train_X_dim_p_list, train_DA_dim_p_list)

		#(3)- 计算g_DA
		g_DA = H_D - H_DA

		return g_DA

	def getInformationGainRatio(self, train_X_list, train_Y_list, train_X_dim):
		'''

		:param train_X_dim: 同get_InformetionGain函数
		:return: 信息增益率
		'''
		#(1)- 计算H_DA
		train_X_dim_list = [train_X[train_X_dim] for train_X in train_X_list]
		train_X_dim_kinds_list = list(set(train_X_dim_list))
		train_DA_dim_p_list = []
		for entropy_count in range(len(train_X_dim_kinds_list)):
			train_Y_divided_list = []
			train_X_item_divided = train_X_dim_kinds_list[entropy_count]
			for train_Y_count in range(len(train_Y_list)):
				if(train_X_dim_list[train_Y_count] == train_X_item_divided):
					train_Y_divided_list.append(train_Y_list[train_Y_count])
			train_DA_Num = len(train_Y_divided_list)
			train_DA_Y_kinds_list = list(set(train_Y_divided_list))
			H_DA_P = [(train_Y_divided_list.count(train_Y) / train_DA_Num) for train_Y in train_DA_Y_kinds_list]
			H_DA_divided = [(p * math.log(1/p,2)) for p in H_DA_P]
			H_DA_divided = sum(H_DA_divided)
			train_DA_dim_p_list.append(H_DA_divided)
		H_AD = sum(train_DA_dim_p_list)

		#(2)- 计算信息增益
		g_DA = self.get_InformetionGain(train_X_list, train_Y_list, train_X_dim)

		#(3)- 计算信息增益比
		if(H_AD != 0):
			gr_DA = g_DA/H_AD
		else:
			gr_DA = 1000000
		return gr_DA

	#1.2- 决策树生成
	def createDT(self, train_X_list, train_X_labels, train_Y_list, DT_method):
		'''

		:param train_X_list:
		:param train_X_labels:
		:param train_Y_list:
		:param DT_method:
		:return: 决策树{}
		'''
		DT = {}

		#(1)- train_Y_list所有值相等，即同一类
		if(train_Y_list.count(train_Y_list[0]) == len(train_Y_list)):
			#print("特殊情况1- train_Y_list所有值相等，即同一类")
			DT = train_Y_list[0]
			return DT
		#(2)- train_X_labels为空集，即无特征
		if(len(train_X_labels) == 0):
			#print("特殊情况2- train_X_labels为空集，即无特征")
			Y_counts = Counter(train_Y_list)
			top_Y = Y_counts.most_common(1)
			DT = Y_counts
			return DT
		#(3)- 获取最优特征Ag
		bestValue, bestFeatureIndex, bestFeatureName = \
			self.chooseBestFeature(train_X_list, train_X_labels, train_Y_list, DT_method)

		#(4)- 判断bestValue是否小于epsilon
		if(bestValue <= self.epsilon):
			#print("特殊情况3- bestValue小于epsilon")
			Y_counts = Counter(train_Y_list)
			top_Y = word_counts.most_common(1)
			DT = Y_counts
			return DT

		# 获得当前bestFeatureName值
		bestFeature_col_list = [ train_X[bestFeatureIndex] for train_X in train_X_list ]
		bestFeature_col_valueKinds_list = list(set(bestFeature_col_list))

		#(5)- 去掉bestFeatureIndex对应的特征，对train_X_list、train_X_labels、train_Y_list进行划分
		new_train_XMatrix_list, new_train_X_labels, new_train_YMatrix_list = \
			self.split_trainSets(train_X_list, train_X_labels, train_Y_list, bestFeatureIndex)

		#(6)- 遍历新的 train_X_list、 train_X_labels、 train_Y_list，递归处理
		DT[bestFeatureName] = {}
		for index in range(len(bestFeature_col_valueKinds_list)):
			bestFeature_col_valueKinds = bestFeature_col_valueKinds_list[index]
			DT[bestFeatureName][bestFeature_col_valueKinds] = \
				self.createDT(new_train_XMatrix_list[index], new_train_X_labels, new_train_YMatrix_list[index], DT_method)

		return DT

	#2- 决策树分类
	def classify(self, DT, text_X, train_X_labels):
		text_X_buffer = copy.deepcopy(text_X)
		train_X_labels_buffer = copy.deepcopy(train_X_labels)

		#1- 先找根节点
		root = [ key for key in DT.keys() ]
		if(len(root) != 1):
			print('DT决策树根节点数目不是1个！！！')
			return None
		root = root[0]
		root_index = train_X_labels_buffer.index(root)
		root_value_in_textX = text_X_buffer[root_index]

		DT_child = DT[root][root_value_in_textX]
		if(type(DT_child).__name__ == 'dict'):
			#plotDT.createPlot(DT_child, '子树')
			pass
		else:
			self.classResult = DT_child
			return
		#2- 递归classsify
		classResult = self.classify(DT_child, text_X_buffer, train_X_labels_buffer)

	def calculateAccuracy(self, DT, text_X_list_withlabel, train_X_labels):
		print('\t\t==决策树分类测试结果==')
		timeCount = 1
		text_X_Num = len(text_X_list_withlabel)
		text_X_RightNum = 0
		for text_X_withlabel in text_X_list_withlabel:
			text_X = text_X_withlabel[0:-1]
			print('第{}次测试: {}'.format(timeCount, text_X))
			self.classify(DT, text_X, train_X_labels)
			print('第{}次分类结果: {}'.format(timeCount, self.classResult))
			timeCount += 1

			if(text_X_withlabel[-1] == self.classResult):
				text_X_RightNum += 1

		text_X_accuracy = round((text_X_RightNum/text_X_Num) * 100,2)
		print('测试集准确率: {}%'.format(text_X_accuracy))
		return text_X_accuracy

	#3- 后剪枝（Failed）
	pass


class CART_DecisionTree(ID3orC45_DecisionTree):
	"""docstring for CART_DecisionTree"""
	def __init__(self, maxDivideCount, Gini_threshold=0.01):
		#(1)- LSRT用到全局变量
		self.LSRT_usedCount = 0
		self.LSRT_maxDivideCount = maxDivideCount

		#(2)- CART用到全局变量
		self.Gini_threshold = Gini_threshold
		self.CART_root = 0
		pathsFile = open('leafNodePaths.txt', 'a', encoding='utf-8')
		self.pathsFile = pathsFile
		self.CART_Tk_list = []

	#1- 最小二乘回归树（least squares regression tree）
	#参考：https://zhuanlan.zhihu.com/p/42505644 示例
	def LSRT(self, LSRT_tree, LSRT_train_X_list, LSRT_train_Y_list, best_c=0):
		#(0)- 递归结束条件
		#超过迭代次数 or 碰到结束节点
		if((self.LSRT_usedCount > self.LSRT_maxDivideCount) or len(LSRT_train_X_list)==1):
			return best_c
		else:
			self.LSRT_usedCount += 1

		#(1)- 遍历train_X所有维度，选取最好(j,s)对（L值最小）
		L_dict = {}	#每个j里面，最好(j,s)对

		for j in range(len(LSRT_train_X_list[0])):
			L_j_dict = {}

			train_X_jDim_value_list = [ train_X[j] for train_X in LSRT_train_X_list ]
			train_X_jDim_value_list.sort()
			s_list = [ round( (train_X_jDim_value_list[index-1]+train_X_jDim_value_list[index])/2, 2 ) \
				for index in range(1,len(train_X_jDim_value_list)) ]

			for s_index, s in enumerate(s_list):
				c1 = sum(LSRT_train_Y_list[0:s_index+1]) / len(LSRT_train_Y_list[0:s_index+1])
				c2 = sum(LSRT_train_Y_list[s_index+1:]) / len(LSRT_train_Y_list[s_index+1:])
				c1 = round(c1, 2)
				c2 = round(c2, 2)

				L_now = sum( np.power((np.array(LSRT_train_Y_list[0:s_index+1]) - c1), 2) ) + \
					sum( np.power((np.array(LSRT_train_Y_list[s_index+1:]) - c2), 2) )
				L_now = round(L_now,2)
				L_j_dict[(j,s)] = [L_now,(c1,c2)]

			L_j_list = sorted(L_j_dict.items(), key=lambda d: d[1][0])	#按L_now排序
			minL_j = L_j_list[0]
			L_dict[minL_j[0]]  = minL_j[1]

		# 求最好(j,s)
		L_dict_sorted = sorted(L_dict.items(), key=lambda d: d[1][0])
		best_js = L_dict_sorted[0][0]	#按L_now排序
		best_c1c2 = L_dict_sorted[0][1][1]
		best_c1c2 = [ round(c,2) for c in best_c1c2 ]	#(2)- 输出c1, c2

		#(2)- 用(j,s)划分区域 R1, R2
		R1 = []
		R2 = []
		R1_Y = []
		R2_Y = []
		for X_index, train_X in enumerate(LSRT_train_X_list):
			if(train_X[best_js[0]] <= best_js[1]):
				R1.append(train_X)
				R1_Y.append(LSRT_train_Y_list[X_index])
			else:
				R2.append(train_X)
				R2_Y.append(LSRT_train_Y_list[X_index])


		#(3)- 对R1，R2递归调用LSRT
		LSRT_tree['x_{} > {}'.format(best_js[0], best_js[1])] = {'否':{}, '是':{}}
		LSRT_tree['x_{} > {}'.format(best_js[0], best_js[1])]['否'] = \
			self.LSRT(LSRT_tree['x_{} > {}'.format(best_js[0], best_js[1])]['否'], R1, R1_Y, best_c1c2[0])
		LSRT_tree['x_{} > {}'.format(best_js[0], best_js[1])]['是'] = \
			self.LSRT(LSRT_tree['x_{} > {}'.format(best_js[0], best_js[1])]['是'], R2, R2_Y, best_c1c2[1])

		#(4)- 返回LSRT树
		return LSRT_tree

	#2- CART树,类似LSRT思路
	def getGiniD(self, D_Y):
		D_Y_kinds = list(set(D_Y))
		D_Y_count = Counter(D_Y)
		p_list = [ D_Y_count[kind]/len(D_Y) for kind in D_Y_kinds ]
		return ( 1 - sum(np.power(np.array(p_list),2)) )

	def CART(self, CART_tree, train_X_list, train_X_labels, train_Y_list, Gini_index = 100000, best_c=0):
		#(0)- 递归结束条件
		#超过迭代次数 or 碰到结束节点
		if((Gini_index < self.Gini_threshold) or len(set(train_Y_list))==1):
			return best_c

		#(1) & (2)- 
		#	遍历train_X_labels，对当前label取值，分割为D1 & D2，计算此时Gini(D|A)
		#		挑选最小Gini(D|A)对应的(A,a)对
		Gini_bestDA_node_list = []
		Gini_bestDA_list = []
		for featureIndex, A in enumerate(train_X_labels):
			Gini_DA_now_list = []
			Gini_node_list = []
			featureValue_kinds_list = [ train_X[featureIndex] for train_X in train_X_list ]
			featureValue_kinds_list = list(set(featureValue_kinds_list))

			for a in featureValue_kinds_list:
				D1 = []
				D2 = []
				D1_Y = []
				D2_Y = []
				for index, train_X in enumerate(train_X_list):
					if(train_X[featureIndex]==a):
						D1.append(train_X)
						D1_Y.append(train_Y_list[index])
					else:
						D2.append(train_X)
						D2_Y.append(train_Y_list[index])

				Gini_DA_now = (len(D1)/len(train_X_list))*self.getGiniD(D1_Y) + \
					(len(D2)/len(train_X_list))*self.getGiniD(D2_Y)
				#Gini_DA_now = round(Gini_DA_now,2)
				Gini_DA_now_list.append(Gini_DA_now)
				Gini_node_list.append({Gini_DA_now:[featureIndex,A,a]})

			Gini_bestDA_list.append(min(Gini_DA_now_list))
			Gini_bestDA_node_list.append(Gini_node_list[Gini_DA_now_list.index(min(Gini_DA_now_list))])

		best_Aa_node = Gini_bestDA_node_list[ Gini_bestDA_list.index(min(Gini_bestDA_list))]
		# print(Gini_bestDA_list,'\n',Gini_bestDA_node_list,'\n',best_Aa_node)
		# {0.27: [2, '有自己的房子', '否']}

		#(3)- 根据best_Aa_node划分 D1 & D2
		next_D1 = []
		next_D2 = []
		next_D1_Y = []
		next_D2_Y = []
		bestFeatureDIndex = list(best_Aa_node.values())[0][0]
		bestFeatureName = list(best_Aa_node.values())[0][1]
		bestFeatureValue = list(best_Aa_node.values())[0][2]
		best_Gini = list(best_Aa_node.keys())[0]

		if(best_c==0):
			self.CART_root = bestFeatureName

		for index, train_X in enumerate(train_X_list):
			if(train_X[bestFeatureDIndex]==bestFeatureValue):
				next_D1.append(train_X)
				next_D1_Y.append(train_Y_list[index])
			else:
				next_D2.append(train_X)
				next_D2_Y.append(train_Y_list[index])

		#(4)- 递归调用
		D1_Y_maxNum_kind = 'None'
		D2_Y_maxNum_kind = 'None'
		if(D1_Y):
			D1_Y_maxNum_kind = Counter(D1_Y).most_common(1)
			D1_Y_maxNum_kind = D1_Y_maxNum_kind[0]
			D1_Y_maxNum_kind = D1_Y_maxNum_kind[0]

		if(D2_Y):
			D2_Y_maxNum_kind = Counter(D2_Y).most_common(1)
			D2_Y_maxNum_kind = D2_Y_maxNum_kind[0]
			D2_Y_maxNum_kind = D2_Y_maxNum_kind[0]

		CART_tree[bestFeatureName] = {'√:{}'.format(bestFeatureValue):{}, \
			'X:{}'.format(bestFeatureValue):{}}
		CART_tree[bestFeatureName]['√:{}'.format(bestFeatureValue)] = \
			self.CART(CART_tree[bestFeatureName]['√:{}'.format(bestFeatureValue)], next_D1, train_X_labels, next_D1_Y, best_Gini, D1_Y_maxNum_kind)
		CART_tree[bestFeatureName]['X:{}'.format(bestFeatureValue)] = \
			self.CART(CART_tree[bestFeatureName]['X:{}'.format(bestFeatureValue)], next_D2, train_X_labels, next_D2_Y, best_Gini, D2_Y_maxNum_kind)

		return CART_tree

	#3- 剪枝（后剪枝）
	def get_CTt(self, aboveTt, train_X_list, train_X_labels, train_Y_list):
		feature_count = len(aboveTt)/2
		train_X_list_aboveTt = []
		train_Y_list_aboveTt = []
		for X_index, train_X in enumerate(train_X_list):
			passFlag = 1
			for featureIndex in range(0,len(aboveTt)-1, 2):
				featureName = aboveTt[featureIndex]
				featureValue_Name = aboveTt[featureIndex+1][2:]
				featureValue_yesORno = aboveTt[featureIndex+1][0]
				if(featureValue_yesORno == '√'):
					if(train_X[train_X_labels.index(featureName)] != featureValue_Name):
						passFlag = 0
						break
				elif(featureValue_yesORno == 'X'):
					if(train_X[train_X_labels.index(featureName)] == featureValue_Name):
						passFlag = 0
						break

			if(passFlag == 1):
				train_X_list_aboveTt.append(train_X)
				train_Y_list_aboveTt.append(train_Y_list[X_index])

		#print('\n==============\n', aboveTt, '\n', train_X_list_aboveTt, '\n', train_Y_list_aboveTt)
		return self.getGiniD(train_Y_list_aboveTt), [Counter(train_Y_list_aboveTt).most_common(1)][0][0]

	def updateTree(self, CART_tree, pruning_aboveTt, mostY):
		if(len(pruning_aboveTt) == 0):
			return mostY
		else:
			if(len(pruning_aboveTt) == 2):
				pruning_aboveTt_buffer = []
			else:
				pruning_aboveTt_buffer = pruning_aboveTt[2:]
				pruning_aboveTt_buffer = copy.deepcopy(pruning_aboveTt_buffer)

			CART_tree[pruning_aboveTt[0]][pruning_aboveTt[1]] = \
			self.updateTree(CART_tree[pruning_aboveTt[0]][pruning_aboveTt[1]]\
				, pruning_aboveTt_buffer, mostY)
			return CART_tree

	def post_pruning(self, CART_tree, train_X_list, train_X_labels, train_Y_list, alpha = 100000000):
		#(0)- 递归退出
		secondTree = CART_tree[list(CART_tree.keys())[0]]
		secondKeys = list(secondTree.keys())
		if(len(secondKeys) == 2 and type(secondTree[secondKeys[0]]).__name__ != 'dict' and \
			type(secondTree[secondKeys[1]]).__name__ != 'dict' ):
			self.CART_Tk_list.append(T_k)
			return

		#(1)&(2)- 设k = 0, T = T_0, alpha = +∞
		T_k = copy.deepcopy(CART_tree)
		self.CART_Tk_list.append(T_k)
		
		#(3)- 对tree_buffer自底向上，计算所有内部节点C(Tt)，|Tt|
		#(3.1)- 统计含有2个叶节点的内部节点
		leafNodePaths_list = self.getleafNodesPaths(T_k, self.CART_root)
		leafNodePaths_seg_list = [ path.split(',') for path in leafNodePaths_list ]
		pathLen_list = [ len(node) for node in leafNodePaths_seg_list]
		pathLen_kinds_list = list(set(pathLen_list))
		samePathLen_list = {}
		samePathLen_list_filter = {}
		for pathLenkind in pathLen_kinds_list:
			samePathLen_list[pathLenkind] = []
			for index, node in enumerate(leafNodePaths_seg_list):
				if(len(node) == pathLenkind):
					samePathLen_list[pathLenkind].append(leafNodePaths_seg_list[index])
		#过滤，不要1的
		for key, value in samePathLen_list.items():
			if(len(value) != 1):
				samePathLen_list_filter[key] = value

		aboveTt_list = []
		Tt_list = []
		for key,value in samePathLen_list_filter.items():
			if(len(value) == 2):
				path1 = value[0]
				path2 = value[1]
				diff_index = 0
				for index, word in enumerate(path1):
					if(path1[index] != path2[index]):
						diff_index = index
						break
				aboveTt_list.append([path1[0:(diff_index-1)], path2[0:(diff_index-1)]])
				Tt_list.append([path1[(diff_index-1):], path2[(diff_index-1):]])
			else:	#剩下情况：除了2的偶数 & 除了1的奇数
				for index1 in range(len(value)):
					for index2 in range(index1+1, len(value)):
						diff_index_count = 0
						diff_index = 0

						path1 = value[index1]
						path2 = value[index2]

						for word_index in range(len(path1)):
							if(path1[word_index] != path2[word_index]):
								diff_index_count += 1

						if(diff_index_count == 2):
							for word_index in range(len(path1)):
								if(path1[word_index] != path2[word_index]):
									diff_index = word_index
									break
							aboveTt_list.append([path1[0:(diff_index-1)], path2[0:(diff_index-1)]])
							Tt_list.append([path1[(diff_index-1):], path2[(diff_index-1):]])

		#(3.2)- 遍历Tt_list计算C(Tt)
		C_Tt_list = []
		C_Tt_mostY_list = []
		for index, Tt in enumerate(Tt_list):
			aboveTt = aboveTt_list[index][0]
			C_Tt_list.append(self.get_CTt(aboveTt, train_X_list, train_X_labels, train_Y_list)[0])
			C_Tt_mostY_list.append(self.get_CTt(aboveTt, train_X_list, train_X_labels, train_Y_list)[1][0])
		#(3.3)- 计算g(t)
		C_t = self.getGiniD(train_Y_list)
		Tt_Num = 2
		gt_list = []
		for C_Tt in C_Tt_list:
			gt = (C_t - C_Tt)/(Tt_Num-1)
			gt_list.append(gt)
		#(3.4)- 得到alpha
		alpha = min(alpha,min(gt_list))

		#(4)- 对gt_list=alpha对应的Tt剪枝
		if(alpha not in gt_list):
			return
		pruningTree_index = gt_list.index(alpha)
		pruning_aboveTt = aboveTt_list[pruningTree_index][0]
		pruning_Tt = Tt_list[pruningTree_index]
		mostY = C_Tt_mostY_list[pruningTree_index]

		#对CART_tree剪枝
		T = self.updateTree(CART_tree, pruning_aboveTt, mostY)
		#plotDT.createPlot(T, '剪枝后 CART决策树')

		#(5)- 递归
		self.post_pruning(T, train_X_list, train_X_labels, train_Y_list, alpha)


def ID3orC45_test():
	print('=============ID3 and C4.5决策树测试=============')

	#0- 数据集准备
	# 《统计学习方法-李航》 P59 表5.1
	datasets1 = [['青年', '否', '否', '一般', '否'],
			   ['青年', '否', '否', '好', '否'],
			   ['青年', '是', '否', '好', '是'],
			   ['青年', '是', '是', '一般', '是'],
			   ['青年', '否', '否', '一般', '否'],
			   ['中年', '否', '否', '一般', '否'],
			   ['中年', '否', '否', '好', '否'],
			   ['中年', '是', '是', '好', '是'],
			   ['中年', '否', '是', '非常好', '是'],
			   ['中年', '否', '是', '非常好', '是'],
			   ['老年', '否', '是', '非常好', '是'],
			   ['老年', '否', '是', '好', '是'],
			   ['老年', '是', '否', '好', '是'],
			   ['老年', '是', '否', '非常好', '是'],
			   ['老年', '否', '否', '一般', '否']
			]
	# 《机器学习-周志华》
	#P76
	datasets2 = [['青绿','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
				['乌黑','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['青绿','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
				['浅白','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['青绿','稍蜷','浊响','清晰','稍凹','软粘','是'],
				['乌黑','稍蜷','浊响','稍糊','稍凹','软粘','是'],
				['乌黑','稍蜷','浊响','清晰','稍凹','硬滑','是'],
				['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑','是'],
				['青绿','硬挺','清脆','清晰','平坦','软粘','否'],
				['浅白','硬挺','清脆','模糊','平坦','硬滑','否'],
				['浅白','蜷缩','浊响','模糊','平坦','软粘','否'],
				['青绿','稍蜷','浊响','稍糊','凹陷','硬滑','否'],
				['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑','否'],
				['乌黑','稍蜷','浊响','清晰','稍凹','软粘','否'],
				['浅白','蜷缩','浊响','模糊','平坦','硬滑','否'],
				['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑','否']
			]
	#P80 训练集
	datasets3 = [['青绿','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
				['乌黑','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['青绿','稍蜷','浊响','清晰','稍凹','软粘','是'],
				['乌黑','稍蜷','浊响','稍糊','稍凹','软粘','是'],
				['青绿','硬挺','清脆','清晰','平坦','软粘','否'],
				['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑','否'],
				['乌黑','稍蜷','浊响','清晰','稍凹','软粘','否'],
				['浅白','蜷缩','浊响','模糊','平坦','硬滑','否'],
				['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑','否']
			]
	#P80 测试集
	text_X_list_withlabel = [['青绿','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
				['浅白','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['乌黑','稍蜷','浊响','清晰','稍凹','硬滑','是'],
				['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑','是'],
				['浅白','硬挺','清脆','模糊','平坦','硬滑','否'],
				['浅白','蜷缩','浊响','模糊','平坦','软粘','否'],
				['青绿','稍蜷','浊响','稍糊','凹陷','硬滑','否']
			]

	train_X_list = [ data[0:-1] for data in datasets3 ]
	#train_X_labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
	train_X_labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
	train_Y_list = [ data[-1] for data in datasets3 ]

	train_X_list_buffer = copy.deepcopy(train_X_list)
	train_X_labels_buffer = copy.deepcopy(train_X_labels)
	train_Y_list_buffer = copy.deepcopy(train_Y_list)

	#2- ID3 C4.5决策树算法测试
	DT_ID3orC45 = ID3orC45_DecisionTree(0.1)
	#2.1- ID3
	DT_ID3 = DT_ID3orC45.createDT(train_X_list, train_X_labels, train_Y_list, 1)
	print("ID3决策树:", DT_ID3)
	plotDT.createPlot(DT_ID3, 'ID3决策树')
	
	# #2.2- C4.5
	DT_C45 = DT_ID3orC45.createDT(train_X_list_buffer, train_X_labels_buffer, train_Y_list_buffer, 2)
	print("C4.5决策树:", DT_C45)
	plotDT.createPlot(DT_C45, 'C4.5决策树')

	#3- 分类
	DT_ID3orC45.calculateAccuracy(DT_ID3, text_X_list_withlabel, train_X_labels_buffer)

	#4- 后剪枝(ID3)
	pass


def CART_test():
	print('=============CART决策树测试=============')
	LSRT_maxDivideCount = 10
	Gini_threshold = 0.001
	CART_DT = CART_DecisionTree(LSRT_maxDivideCount, Gini_threshold)

	#1- LSRT
	#(1)- 输入数据
	# 测试data来源: https://zhuanlan.zhihu.com/p/42505644
	LSRT_train_X_list = list(range(1,11))
	for x_index, x in enumerate(LSRT_train_X_list):
		if(type(x).__name__ != 'list'):
			LSRT_train_X_list[x_index] = [x]
	LSRT_train_Y_list = [5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05]

	#(2)- LSRT
	LSRT_tree = CART_DT.LSRT({}, LSRT_train_X_list, LSRT_train_Y_list)
	print(LSRT_tree)
	plotDT.createPlot(LSRT_tree, 'LSRT决策树\n参数:{}'.format(LSRT_maxDivideCount))

	#2- CART
	#(1)- 输入数据
	datasets1 = [['青年', '否', '否', '一般', '否'],
		   ['青年', '否', '否', '好', '否'],
		   ['青年', '是', '否', '好', '是'],
		   ['青年', '是', '是', '一般', '是'],
		   ['青年', '否', '否', '一般', '否'],
		   ['中年', '否', '否', '一般', '否'],
		   ['中年', '否', '否', '好', '否'],
		   ['中年', '是', '是', '好', '是'],
		   ['中年', '否', '是', '非常好', '是'],
		   ['中年', '否', '是', '非常好', '是'],
		   ['老年', '否', '是', '非常好', '是'],
		   ['老年', '否', '是', '好', '是'],
		   ['老年', '是', '否', '好', '是'],
		   ['老年', '是', '否', '非常好', '是'],
		   ['老年', '否', '否', '一般', '否']
		]
	datasets2 = [['青绿','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
				['乌黑','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['青绿','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
				['浅白','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['青绿','稍蜷','浊响','清晰','稍凹','软粘','是'],
				['乌黑','稍蜷','浊响','稍糊','稍凹','软粘','是'],
				['乌黑','稍蜷','浊响','清晰','稍凹','硬滑','是'],
				['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑','是'],
				['青绿','硬挺','清脆','清晰','平坦','软粘','否'],
				['浅白','硬挺','清脆','模糊','平坦','硬滑','否'],
				['浅白','蜷缩','浊响','模糊','平坦','软粘','否'],
				['青绿','稍蜷','浊响','稍糊','凹陷','硬滑','否'],
				['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑','否'],
				['乌黑','稍蜷','浊响','清晰','稍凹','软粘','否'],
				['浅白','蜷缩','浊响','模糊','平坦','硬滑','否'],
				['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑','否']
			]
	
	#P80 训练集
	datasets3 = [['青绿','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
				['乌黑','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['青绿','稍蜷','浊响','清晰','稍凹','软粘','是'],
				['乌黑','稍蜷','浊响','稍糊','稍凹','软粘','是'],
				['青绿','硬挺','清脆','清晰','平坦','软粘','否'],
				['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑','否'],
				['乌黑','稍蜷','浊响','清晰','稍凹','软粘','否'],
				['浅白','蜷缩','浊响','模糊','平坦','硬滑','否'],
				['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑','否']
			]

	#P80 测试集
	text_X_list_withlabel = [['青绿','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
				['浅白','蜷缩','浊响','清晰','凹陷','硬滑','是'],
				['乌黑','稍蜷','浊响','清晰','稍凹','硬滑','是'],
				['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑','是'],
				['浅白','硬挺','清脆','模糊','平坦','硬滑','否'],
				['浅白','蜷缩','浊响','模糊','平坦','软粘','否'],
				['青绿','稍蜷','浊响','稍糊','凹陷','硬滑','否']
			]

	train_X_list = [ data[0:-1] for data in datasets2 ]
	#train_X_labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
	train_X_labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
	train_Y_list = [ data[-1] for data in datasets2 ]
	
	#(2)- CART
	CART_tree = CART_DT.CART({}, train_X_list, train_X_labels, train_Y_list)
	print(CART_tree)
	plotDT.createPlot(CART_tree, 'CART决策树\nGini参数:{}'.format(Gini_threshold))

	#(3)- 分类
	pass

	#(4)- (后)剪枝
	CART_DT.post_pruning(CART_tree, train_X_list, train_X_labels, train_Y_list)
	[plotDT.createPlot(tree, 'CART决策树\n（剪枝后）')\
	 for tree in CART_DT.CART_Tk_list[1:]]



if __name__ == '__main__':
	#1- ID3 and C4.5决策树测试
	ID3orC45_test()
	
	#2- CART决策树测试
	CART_test()
