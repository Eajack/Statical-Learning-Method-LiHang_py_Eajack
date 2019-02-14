'''
!/usr/bin/env python3
-*- coding : utf-8 -*-
Author: Eajack
date:2019/1/24
Function：
	SMO简化版算法实现，
	参考
		1- 《机器学习实战》
		2- https://www.cnblogs.com/nolonely/p/6541527.html
'''
import numpy as np
import random

def calcKernelValue(train_X_mat, train_X_i, kernelOption):
    kernelType = kernelOption[0]
    numSamples = train_X_mat.shape[0]
    kernelValue = np.mat(np.zeros((numSamples, 1)))
    
    if kernelType == 'linear':
        kernelValue = train_X_mat * train_X_i.T
    elif kernelType == 'rbf':
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(numSamples):
            diff = train_X_mat[i, :] - train_X_i
            kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma**2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return kernelValue

def select_Jrand(i,trainNum):
	j = i
	while(j == i):
		j = int(random.uniform(0,trainNum))
	return j

def clipAlpha(aj, maxValue, minValue):
	if aj > maxValue:
		aj = maxValue
	if aj < minValue:
		aj = minValue
	return aj

def SMO_simple(train_X_list, train_Y_list, C, toler, maxLoopTime, kernelOption):
	train_X_mat = np.mat(train_X_list)
	train_Y_mat = np.mat(train_Y_list).transpose()
	b = 0
	train_X_num, train_X_dim = np.shape(train_X_mat)
	alphas = np.mat(np.zeros((train_X_num,1)))
	iterCount = 0

	while(iterCount < maxLoopTime):
		print('+++++++++++++++第{}次循环++++++++++++++++'.format(iterCount))
		alphaPairsChanged = 0
		for i in range(train_X_num):	#外层循环，第一个变量i
			print('=====外层训练{}====='.format(i))
			g_xi = float( np.multiply(alphas,train_Y_mat).T * \
				calcKernelValue(train_X_mat,train_X_mat[i,:],kernelOption) ) + b
			E_i = g_xi - float(train_Y_mat[i])
			# 1- 第一层变量选择，KKT条件
			## satisfy KKT condition
		    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
		    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
		    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
		    ## violate KKT condition
		    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
		    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct) 
		    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
		    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
			# KKT_satisfy = (
			# 	((alphas[i]==0) and (train_Y_mat[i]*g_xi >= 1)) and
			# 	((alphas[i]>0 and alphas[i]<C) and (train_Y_mat[i]*g_xi == 1)) and
			# 	((alphas[i]==C) and (train_Y_mat[i]*g_xi <= 1))
			# )
			KKT_violate = (
					((train_Y_mat[i] * E_i < -toler) and (alphas[i] < C)) or 
					((train_Y_mat[i] * E_i > toler) and (alphas[i] > 0))
				)

			if(KKT_violate == True):
				# 2- 不符合KKT，开始选择第二层变量j
				j = select_Jrand(i, train_X_num)
				g_xj = float( np.multiply(alphas,train_Y_mat).T * \
					calcKernelValue(train_X_mat,train_X_mat[j,:],kernelOption) ) + b
				E_j = g_xj - float(train_Y_mat[j])

				alphaI_old = alphas[i].copy()
				alphaJ_old = alphas[j].copy()

				# 保证alpha在0~C之间
				if(train_Y_mat[i] != train_Y_mat[j]):
					minValue = max(0, alphas[j] - alphas[i])
					maxValue = min(C, C+alphas[j]-alphas[i])
				else:
					minValue = max(0, alphas[j]+alphas[i]-C)
					maxValue = min(C, alphas[j]+alphas[i])
				if(minValue == maxValue):
					print('minValue = maxValue')
					continue

				#3- 对alphas[i] & alphas[j]进行更新
				eta = 2.0*train_X_mat[i,:]*train_X_mat[j,:].T - \
					train_X_mat[i,:]*train_X_mat[i,:].T - \
					train_X_mat[j,:]*train_X_mat[j,:].T
				if(eta >= 0):
					print('eta >= 0')
					continue
				alphas[j] -= (train_Y_mat[j]*(E_i - E_j)/eta).tolist()[0][0]
				alphas[j] = clipAlpha(alphas[j], maxValue, minValue)
				if(abs(alphas[j] - alphaJ_old) < 0.00001):
					print('J not move enough')
					continue
				alphas[i] += (train_Y_mat[j]*train_Y_mat[i]*(alphaJ_old - alphas[j])).tolist()[0][0]

				#4- 更新阈值b & 差值E_i
				b1 = b - E_i - train_Y_mat[i]*(alphas[i]-alphaI_old)*\
					train_X_mat[i,:]*train_X_mat[i,:].T - \
					train_Y_mat[j]*(alphas[j]-alphaJ_old)*\
					train_X_mat[i,:]*train_X_mat[j,:].T 
				b2 = b - E_j - train_Y_mat[i]*(alphas[i]-alphaI_old)*\
					train_X_mat[i,:]*train_X_mat[j,:].T - \
					train_Y_mat[j]*(alphas[j]-alphaJ_old)*\
					train_X_mat[j,:]*train_X_mat[j,:].T
				if(0 < alphas[i]) and (alphas[i] < C):
					b = b1
				elif(0 < alphas[j]) and (alphas[j] < C):
					b = b2
				else:
					b = (b1+b2)/2.0
				
				alphaPairsChanged += 1
				print('iterCount:{},i:{},pairs changed:{}'.format(iterCount,i,alphaPairsChanged))

		if(alphaPairsChanged == 0):
			iterCount += 1
		else:
			iterCount = 0
		print('iteration number:{}'.format(iterCount))

	
	return alphas, b