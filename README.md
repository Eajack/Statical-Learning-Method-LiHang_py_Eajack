# Statical-Learning-Method-LiHang_py_Eajack

## 1、运行环境
> * Windows or Linux
> * Python3.5.2(Python 3.x.x)

## 2、第三方库汇总
> * matplotlib
> * numpy
> * sklearn
>* pandas

## 3、参考资料
> * 《统计学习方法-李航》
> * 《机器学习实战》
> * Google

## 4、代码说明
只用py实现第1章-第8章，第9章-第11章，由于个人时间、章节难度&意志（说白了，就是看到第8章时，刚好隔了一段时间玩了，重新捡起时不想看了…）原因，未完成py实现（之前想的C/C++版就…算了吧）

大致介绍下每章节代码（部分章节的部分内容，未实现）

1. [Chapter_1](https://github.com/Eajack/Statical-Learning-Method-LiHang_py_Eajack/tree/master/Chapter_1)
>* 最小二乘法（Least Square Method）的单变量&多变量版本
>* 正则化L2
>* PS：正则化L1未实现
2. [Chapter_2](https://github.com/Eajack/Statical-Learning-Method-LiHang_py_Eajack/tree/master/Chapter_2)
>* 感知机（Perceptron）的原始形式&对偶形式
3. [Chapter_3](https://github.com/Eajack/Statical-Learning-Method-LiHang_py_Eajack/tree/master/Chapter_3)
>* KNN（线性扫描&kd树版本）
4. [Chapter_4](https://github.com/Eajack/Statical-Learning-Method-LiHang_py_Eajack/tree/master/Chapter_4)
>* 朴素贝叶斯（Naive Bayes）
>* 贝叶斯估计（lambda = 1，又称“拉普拉斯平滑”，Laplace Smoothing）
5. [Chapter_5](https://github.com/Eajack/Statical-Learning-Method-LiHang_py_Eajack/tree/master/Chapter_5)
>* ID3决策树，生成、分类
>* C4.5决策树，生成、分类
>* CART决策树，生成、分类、剪枝
6. [Chapter_6](https://github.com/Eajack/Statical-Learning-Method-LiHang_py_Eajack/tree/master/Chapter_6)
>* Logistic回归
>* 梯度下降法（Gradient Descent,GD）：随机GD（stochastic，SGD）、批量GD（Batch GD）、小批量GD（mini-batch，MBGD）
>* PS：本章还有最大熵模型、拟牛顿法&改进的迭代尺度法（imporoved iterative scaling，IIS）未实现
7. [Chapter_7](https://github.com/Eajack/Statical-Learning-Method-LiHang_py_Eajack/tree/master/Chapter_7)
>* 线性可分SVM（Linear Support Vector Machine in Linearly Separable Case，LSVML）
>* 序列最小最优化算法（Sequence Minimal Optimization，SMO）
>* PS：SVM还包括线性SVM、非线性SVM，未实现。与LSVML思路大致相同，不同于前二者分别引入惩罚系数C、惩罚系数C&核函数
8. [Chapter_8](https://github.com/Eajack/Statical-Learning-Method-LiHang_py_Eajack/tree/master/Chapter_8)
>* Adaboost
>* PS：未实现提升树
9. Chapter_9 ~ Chapter_11：
>* EM&GEM
>* 隐马尔科夫模型（只略看数学原理）
>* 条件随机场（未看）
10. [Extra-MLinAction](https://github.com/Eajack/Statical-Learning-Method-LiHang_py_Eajack/tree/master/Extra-MLinAction)
>* Kmeans（《机器学习方法》挑的实现章节）

## 5、最后一点
代码仅供参考，质量一般。这书若每天腾空专心看，大致需耗时2个月可看完，并py代码实现。（C/C++就不一定了，矩阵运算等库可能没py方便）
