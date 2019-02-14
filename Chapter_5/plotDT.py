# 来源：https://blog.csdn.net/sinat_29957455/article/details/76553987
import matplotlib.pyplot as plt

#设置画节点用的盒子的样式
decisionNode = dict(boxstyle = "sawtooth",fc="0.8")
leafNode = dict(boxstyle = "round4",fc="0.8")
#设置画箭头的样式    http://matplotlib.org/api/patches_api.html#matplotlib.patches.FancyArrowPatch
arrow_args = dict(arrowstyle="<-")
#用来正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']
#用来正常显示负号
plt.rcParams['axes.unicode_minus']=False


def getNumLeafs(myTree):
    #初始化树的叶子节点个数
    numLeafs = 0
    #myTree.keys()获取树的非叶子节点'no surfacing'和'flippers'
    #list(myTree.keys())[0]获取第一个键名'no surfacing'
    firstStr = list(myTree.keys())[0]
    #通过键名获取与之对应的值，即{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    secondDict = myTree[firstStr]
    #遍历树，secondDict.keys()获取所有的键
    for key in secondDict.keys():
        #判断键是否为字典，键名1和其值就组成了一个字典，如果是字典则通过递归继续遍历，寻找叶子节点
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        #如果不是字典，则叶子结点的数目就加1
        else:
            numLeafs += 1
    #返回叶子节点的数目
    return numLeafs

def getTreeDepth(myTree):
    #初始化树的深度
    maxDepth = 0
    #获取树的第一个键名
    firstStr = list(myTree.keys())[0]
    #获取键名所对应的值
    secondDict = myTree[firstStr]
    #遍历树
    for key in secondDict.keys():
        #如果获取的键是字典，树的深度加1
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        #去深度的最大值
        if thisDepth > maxDepth : maxDepth = thisDepth
    #返回树的深度
    return maxDepth

#绘图相关参数的设置
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    #annotate函数是为绘制图上指定的数据点xy添加一个nodeTxt注释
    #nodeTxt是给数据点xy添加一个注释，xy为数据点的开始绘制的坐标,位于节点的中间位置
    #xycoords设置指定点xy的坐标类型，xytext为注释的中间点坐标，textcoords设置注释点坐标样式
    #bbox设置装注释盒子的样式,arrowprops设置箭头的样式
    '''
    figure points:表示坐标原点在图的左下角的数据点
    figure pixels:表示坐标原点在图的左下角的像素点
    figure fraction：此时取值是小数，范围是([0,1],[0,1]),在图的左下角时xy是（0,0），最右上角是(1,1)
    其他位置是按相对图的宽高的比例取最小值
    axes points : 表示坐标原点在图中坐标的左下角的数据点
    axes pixels : 表示坐标原点在图中坐标的左下角的像素点
    axes fraction : 与figure fraction类似，只不过相对于图的位置改成是相对于坐标轴的位置
    '''
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,\
    xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',\
    va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
 
#绘制线中间的文字(0和1)的绘制
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]   #计算文字的x坐标
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]   #计算文字的y坐标
    createPlot.ax1.text(xMid,yMid,txtString)
#绘制树
def plotTree(myTree,parentPt,nodeTxt):
    #获取树的叶子节点
    numLeafs = getNumLeafs(myTree)
    #获取树的深度
    depth = getTreeDepth(myTree)
    #firstStr = myTree.keys()[0]
    #获取第一个键名
    firstStr = list(myTree.keys())[0]
    #计算子节点的坐标
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,\
              plotTree.yoff)
    #绘制线上的文字
    plotMidText(cntrPt,parentPt,nodeTxt)
    #绘制节点
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    #获取第一个键值
    secondDict = myTree[firstStr]
    #计算节点y方向上的偏移量，根据树的深度
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #递归绘制树
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            #更新x的偏移量,每个叶子结点x轴方向上的距离为 1/plotTree.totalW
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            #绘制非叶子节点
            plotNode(secondDict[key],(plotTree.xoff,plotTree.yoff),\
                     cntrPt,leafNode)
            #绘制箭头上的标志
            plotMidText((plotTree.xoff,plotTree.yoff),cntrPt,str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD
 
#绘制决策树，inTree的格式为{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
def createPlot(inTree, title):
    #新建一个figure设置背景颜色为白色
    fig = plt.figure(1,facecolor='white')
    #清除figure
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    #创建一个1行1列1个figure，并把网格里面的第一个figure的Axes实例返回给ax1作为函数createPlot()
    #的属性，这个属性ax1相当于一个全局变量，可以给plotNode函数使用
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    #获取树的叶子节点
    plotTree.totalW = float(getNumLeafs(inTree))
    #获取树的深度
    plotTree.totalD = float(getTreeDepth(inTree))
    #节点的x轴的偏移量为-1/plotTree.totlaW/2,1为x轴的长度，除以2保证每一个节点的x轴之间的距离为1/plotTree.totlaW*2
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.text(0.1, 1, title, fontsize=15, ha='left', wrap=True)
    plt.show()
