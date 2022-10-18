from math import log
import operator

"""
函数说明:创建测试数据集
Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 分类属性
年龄：0代表青年，1代表中年，2代表老年；
有工作：0代表否，1代表是；
有自己的房子：0代表否，1代表是；
信贷情况：0代表一般，1代表好，2代表非常好；
类别(是否给贷款)：no代表否，yes代表是。
"""


def createDataSet():
    # 数据集
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    # labels = ['放贷', '不放贷']
    labels = ['年龄', '有工作', '有自己房子', '信贷情况']
    return dataSet, labels


"""
函数说明:计算给定数据集的经验熵(香农熵)
Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
"""


def calcShannonEnt(dataSet):
    # 返回数据集的行数
    numEntires = len(dataSet)
    # 保存每个标签出现次数的字典
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataSet:
        # 提取标签信息
        currentLabel = featVec[-1]
        # 如果标签没有放入统计次数的字典，添加
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 标签计数
        labelCounts[currentLabel] += 1
    # 经验熵
    shannonEnt = 0.0
    # 计算香农熵
    for key in labelCounts:
        # 选择标签的概率
        prob = float(labelCounts[key]) / numEntires
        # 利用公式计算
        shannonEnt -= prob * log(prob, 2)
    # 返回经验熵
    return shannonEnt


"""
函数说明:按照给定特征划分数据集

Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
"""


def splitDataSet(dataSet, axis, value):
    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉axis特征
            reducedFeatVec = featVec[:axis]
            # 将符合条件的添加到返回的数据集
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    # 返回划分后的数据集
    return retDataSet


"""
函数说明:选择最优特征

Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
"""


def chooseBestFeatureToSplit(dataSet):
    # 特征数量
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        # 创建set集合{},元素不可重复
        uniqueVals = set(featList)
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 打印每个特征的信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        # 计算信息增益
        if (infoGain > bestInfoGain):
            # 更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain
            # 记录信息增益最大的特征的索引值
            bestFeature = i
    # 返回信息增益最大的特征的索引值
    return bestFeature


"""
函数说明:统计classList中出现此处最多的元素(类标签)

Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)
"""


def majorityCnt(classList):
    classCount = {}
    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回classList中出现次数最多的元素
    return sortedClassCount[0][0]


"""
函数说明:创建决策树

Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
"""


def createTree(dataSet, labels, featLabels):
    # 取分类标签（是否放贷：yes or no）
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的标签
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return majorityCnt(classList)
    # 选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 最优特征的标签
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    # 根据最优特征的标签生成树
    myTree = {bestFeatLabel: {}}
    # 删除已经使用特征标签
    del (labels[bestFeat])
    # 得到训练集中所有最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]
    # 去掉重复的属性值
    uniqueVals = set(featValues)
    # 遍历特征，创建决策树
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
    return myTree


"""
函数说明:获取决策树叶子结点的数目

Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
"""

def getNumLeafs(myTree):
    # 初始化叶子
    numLeafs = 0
    # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]，
    # 可以使用list(myTree.keys())[0]的方法获取结点属性
    firstStr = next(iter(myTree))
    # 获取下一组字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
    # 测试该接点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs


"""
函数说明:获取决策树的层数
Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
"""
def getTreeDepth(myTree):
    # 初始化决策树深度
    maxDepth = 0
    firstStr = next(iter(myTree))
    # 获取下一个字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
    # 测试该接点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1+ getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth>maxDepth:
            maxDepth = thisDepth
    # 更新层数
    return maxDepth

"""
函数说明:绘制结点

Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
"""


if __name__ == '__main__':
    # dataSet, features = createDataSet()
    # print(dataSet)
    # print(calcShannonEnt(dataSet))
    # print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    dataSet, labels = createDataSet()
    feaLabels = []
    myTree = createTree(dataSet, labels, feaLabels)
    print(myTree)
