# coding: utf-8
from __future__ import print_function
import numpy as np
from power_demand import rnn_data

import matplotlib.pyplot as plt
import math
import random


'''

本代码文件是利用误差数据来进行分类的，属于LSTM+Bayes的Bayes预测部分
1.根据之前power_demand得到的误差数据集，连接训练集和测试集误差数据集。
2.再将纯误差数据和误差数据的标签合并，同时将训练的真实电力数据集和测试的电力数据集合并。
3.为了使得模型更加健壮，因此需要打乱数据集，所以接着获取误差数据集大小的一个全排列索引。
4.与此同时让索引以相同的方式打乱误差数据集和真实的电力数据集，是为了接下来的让误差数据集与真实电量数据集同步绘图。
5.然后从误差数据集中分离训练集和测试集。
6.接着以每个时间点的误差当做属性，利用训练集的样本构建构建每个属性的标准差和均值。
7.然后利用标准差和均值构建高斯分布概率密度函数。
8.再根据高斯密度函数计算每个样本属于某个类别的概率，根据属性的标准差和均值来计算概率，再累乘，得到属于某个类别的概率。

'''



# 导入之前预测结果的数据集
def loadDataSet(filename):
	f = open(filename)
	result = []
	for line in f.readlines():
	    line = line.strip().split(' ')
	    line = [float(i) for i in line]
	    result.append(line)
	result = np.array(result)
	return result


# 构建误差数据集
def createErrDataset():
	# 真实的电力数据集
	_, train_y, _, test_y = rnn_data('./power_data.txt')
	# for i in range(len(test_y)):
	# 	plt.plot(test_y[i])
	# 	plt.show()
	# 预测的电力数据集
	train_predict = loadDataSet('./train_result2.txt')
	test_predict = loadDataSet('./test_result2.txt')
	# 真实与预测的误差
	error_normal = np.abs(train_predict - train_y)
	error_abnormal = np.abs(test_predict - test_y)
	# 连接两个误差数据集
	error = np.concatenate((error_normal, error_abnormal), axis=0)
	print(error.shape)
	# for i in range(51):
	# 	plt.plot(error[i])
	# 	plt.show()

	# 误差数据集的标签
	error_label = [1 for i in range(44)] + [0 for i in range(7)]
	error_label = np.array(error_label).reshape([len(error_label), 1])
	# 将纯误差数据和误差数据的标签合并
	errDataset = np.concatenate((error, error_label), axis=1)
	print(errDataset.shape)
	print(train_y.shape)  # (44,83)
	# 将训练的真实电力数据集和测试的电力数据集合并
	powerDataset = np.concatenate((train_y, test_y), axis=0)
	print(powerDataset.shape)	# (51,83)
	# 获取误差数据集大小的一个全排列
	# 作为索引以相同的方式打乱误差数据集和真实的电力数据集
	index = np.random.permutation(len(errDataset))
	errDataset = errDataset[index]
	powerDataset = powerDataset[index]
	return errDataset, powerDataset


# 从误差数据集中分离训练集和测试集
def splitErrDataset(dataset, ratio=0.8):
	train_nums = int(len(dataset)*ratio)
	trainset = dataset[0:train_nums]
	testset = dataset[train_nums:len(dataset)]
	return trainset, testset


def mean(numbers):
	return sum(numbers)/float(len(numbers))


def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg, 2) for x in numbers]) / float(len(numbers)-1)  # 注意我们使用N-1的方法（译者注：参见无偏估计），也就是在在计算方差时，属性值的个数减1。
	return math.sqrt(variance)

# 根据均值和标准差建立高斯概率密度分布模型
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean, 2) / (2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# 包含数据集中每个属性的均值和标准差
def summarize(dataset):
	summaries =[(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	# 注：在函数调用中使用*list/tuple的方式表示将list/tuple分开，作为位置参数传递给对应函数（前提是对应函数支持不定个数的位置参数）
	# 删除最后的标签
	del summaries[-1]
	return summaries


# 按类别划分数据
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if vector[-1] not in separated:
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated



# 按类别提取属性特征
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	# for i in range(len(separated)):
	# 	attribute = separated[i]
	# 	summaries = summarize(attribute)
	# 	separated[i] = summaries
	summaries = {}
	for classValue, attrset in separated.iteritems():
		summaries[classValue] = summarize(attrset)
	return summaries


# 根据高斯密度函数计算每个样本属于某个类别的概率
# 根据属性的标准差和均值来计算概率，再累乘，得到属于某个类别的概率
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	detail_prob = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		detail_prob[classValue] = []
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			# get every attribute probability value to search the max attribute
			detail_prob[classValue].append(calculateProbability(x, mean, stdev))
			probabilities[classValue] *= calculateProbability(x, mean, stdev)

	# print detail_prob
	return probabilities, detail_prob


# 对单个样本的类别进行预测
def predict(summaries, inputVector):
	probabilities, _ = calculateClassProbabilities(summaries, inputVector)
	# print probabilities
	bestProb = -1
	for classValue, probability in probabilities.iteritems():
		if probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


# 对整个测试集的类别进行预测
def getPreditions(summaries, testset):
	predictions = []
	for i in range(len(testset)):
		result = predict(summaries, testset[i])		# 这里没有写成这样testset[:-1]，是因为之后使用属性来求高斯概率遍历不到的尾部的标签
		predictions.append(result)
	return predictions


# 计算精度
def getAccuracy(testset, predictions):
	correct = 0
	for i in range(len(testset)):
		if testset[i][-1] == predictions[i]:
			correct += 1
	accuracy = correct / float(len(testset)) * 100.0
	return accuracy



# def detectErrLocate(predictions, summaries, testset):
# 	for i in range(len(predictions)):
# 		if predictions[i] == 0:
#			# 获取到详细的每个属性的概率，detail_prob是一个字典，分别是正常类和异常类下属性的概率
# 			_, detail_prob = calculateClassProbabilities(summaries, testset[i])
# 			abnormal_prob = detail_prob[0]
# 			day1 = abnormal_prob[0:11]
# 			otherDays = [abnormal_prob[k:k+12] for k in xrange(11, 83, 12)]
# 			day1_value = sum(day1)
# 			otherDays_value = [sum(j) for j in otherDays]
# 			week_value = [day1_value] + otherDays_value
# 			print week_value
# 			maxDayValue = week_value[0]
# 			errorDay = 1
# 			for j in xrange(len(week_value)):
# 				if week_value[j] > maxDayValue:
# 					maxDayValue = week_value[j]
# 					errorDay = j + 1
# 			print maxDayValue, errorDay
# 	return errorDay


# 检测到误差的位置
# 1.首先获取到预测值
# 2.从预测值中选取预测成异常的类进行处理
# 3.然后根据预测为异常类的角标找到相应的测试集
# 4.对相应的测试集进行处理，即从一周的误差中找到最大的误差
# 5.当然得首先将83个时间点的数据进行处理成一周7天的数据
# 6.再找到最大的一天的数据，即是误差的位置
def detectErrLocate(predictions, summaries, testset):
	# 定义一个误差位置的列表，包括多个样本分别错误的位置	
	errLocate = []
	for i in range(len(predictions)):
		if predictions[i] == 0:
			detectErr = testset[i,:-1]	# 去掉尾部的标签
			# 第一天是11个时间点的数据
			day1 = detectErr[0:11]
			# 12是因为83周期的数据，其实是周期84（因为处理成深度学习训练的标签，所以第一个时间点没有），所以每天是84/7=12个时间点的数据
			otherDays = [detectErr[k:k+12] for k in xrange(11, 83, 12)]
			# 将第一天所产生的电量数据误差合并，或许可以采取求平均数？			
			day1_value = sum(day1)
			otherDays_value = [sum(j) for j in otherDays]
			# 使用‘+’对列表进行合并，而不是extend()方法
			week_value = [day1_value] + otherDays_value
			# 设置最大值，然后循环比较
			maxDayValue = week_value[0]
			errorDay = 1
			for j in range(len(week_value)):
				if week_value[j] > maxDayValue:
					maxDayValue = week_value[j]
					errorDay = j + 1
			print(maxDayValue, errorDay)
			errLocate.append(errorDay)
	return errLocate


# 绘制真实电力和误差数据的图像，只绘制异常的
def plotPowerData(predictions, powerTest, testset):
	for i in range(len(predictions)):
		if predictions[i] == 0:
			fig = plt.figure()
			ax1 = fig.add_subplot(211)	# 211意思是创建2行的1列的图片，这是第1张
			ax2 = fig.add_subplot(212)	# 212意思是创建2行的1列的图片，这是第2张
			ax1.plot(powerTest[i])
			ax2.plot(testset[i,:-1])
			plt.show()


if __name__ == '__main__':
	# 获取误差数据集和电力数据集
	errDataset, powerDataset = createErrDataset()
	for i in range(errDataset.shape[1]-1):
		print(errDataset[:,i])
		plt.hist(errDataset[:,i])
		plt.show()

	print(errDataset.shape)
	print(powerDataset.shape)
	trainset, testset = splitErrDataset(errDataset, 0.67)
	# 获取数据集中每个属性的信息（均值，标准差）	
	summaries = summarizeByClass(trainset)
	
	predictions = getPreditions(summaries, testset)
	print('truly:',testset[:,-1])
	print('predict:',predictions)
	accuracy = getAccuracy(testset, predictions)
	print('Accuracy: {0}%').format(accuracy)
	errLocate = detectErrLocate(predictions, summaries, testset)
	print(errLocate)
	_, powerTest = splitErrDataset(powerDataset, 0.67)
	# plotPowerData(predictions, powerTest, testset)

