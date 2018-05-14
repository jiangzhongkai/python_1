# coding: utf-8
import numpy as np
import power_demand4 as pod4
import matplotlib.pyplot as plt
import math
import random
from sklearn.metrics import roc_curve

'''

本代码文件是利用误差数据来进行分类的，属于LSTM+Bayes的Bayes预测部分
1.根据之前power_demand4得到的误差数据集，连接训练集和测试集误差数据集。
2.再将纯误差数据和误差数据的标签合并，同时将训练的正常车辆数据集和测试的异常车辆数据集合并。
3.为了使得模型更加健壮，因此需要打乱数据集，所以接着获取误差数据集大小的一个全排列索引。
4.与此同时让索引以相同的方式打乱误差数据集和传感器收集到的数据集。
5.然后从误差数据集中分离训练集和测试集。
6.接着以每个时间点的误差当做属性，利用训练集的样本构建构建每个属性的标准差和均值。
7.然后利用标准差和均值构建高斯分布概率密度函数。
8.再根据高斯密度函数计算每个样本属于某个类别的概率，根据属性的标准差和均值来计算概率，再累乘，得到属于某个类别的概率。


'''


# 由于之前输入模型时，前面timestep个数据并没有输入到模型中，被舍去了，
# 故而现在在恢复预测结果时，也要删去剩下的（一个序列本来的长度-timestep）个数据，
# 保持真实数据与预测数据同步
def data_recovery(dataset, timesteps=10, length=18):
	dataset = dataset[length - timesteps:]
	dataset = dataset.reshape([-1,length])
	return dataset


'''
# 由于之前输入模型时，前面timestep个数据并没有输入到模型中，被舍去了，
# 故而现在在恢复预测结果时，也要删去剩下的（一个序列本来的长度-timestep）个数据，
# 保持真实数据与预测数据同步
# 接下来把数据进行逆归一化，还原成数据本来的样子
'''
def inverseNorm(data, norm, timesteps=20):
	data = data[84-20:]
	data = data.reshape([-1,84])
	# print norm['MaxMin']
	data_inversed = data * norm['MaxMin'][1:] + norm['Min'][1:]
	return data_inversed



# 构建误差数据集
def createErrDataset():
	_, train_y, _, test_y, train_norm, test_norm = pod4.rnn_data('./power_data.txt')
	# 拿到之前模型的预测结果，train_result表示为训练集输入模型得到的预测结果
	train_result = np.loadtxt('./train_result3.txt')
	test_result = np.loadtxt('./test_result3.txt')

	train_true = inverseNorm(train_y, train_norm)
	train_predict = inverseNorm(train_result, train_norm)

	test_true = inverseNorm(test_y, test_norm)
	test_predict = inverseNorm(test_result, test_norm)
	# 真实与预测的误差
	error_normal = np.abs(train_true - train_predict)
	error_abnormal = np.abs(test_true - test_predict)
	error_abnormal = np.tile(error_abnormal, [4,1])
	print error_abnormal.shape
	# 连接两个误差数据集
	error = np.concatenate((error_normal, error_abnormal), axis=0)
	print error.shape
	# 是否为异常的标签，1代表是异常
	exception_label = [0 for i in range(len(error_normal))] + [1 for i in range(len(error_abnormal))]
	exception_label = np.array(exception_label).reshape([len(exception_label), 1])
	# 将纯误差数据和是否异常的标签合并
	error_dataset = np.concatenate((error, exception_label), axis=1)
	print error_dataset.shape
	# # 将训练的正常数据集和测试的异常数据集合并
	# power_data = np.concatenate((train_true, test_true), axis=0)
	# # 再将的总的数据集与异常标签合并
	# power_data = np.concatenate((power_data, exception_label), axis=1)
	# print power_data.shape	# (51,83)

	return error_dataset, exception_label


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
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)

	# print detail_prob
	return probabilities


# 对单个样本的类别进行预测
def predict(summaries, inputVector):
	probabilities, _ = calculateClassProbabilities(summaries, inputVector)
	bestProb = -1
	for classValue, probability in probabilities.iteritems():
		if probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel



# 对单个样本的类别进行预测
def predict2(summaries, inputVector, prob_y1):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	prob_y0 = 1 - prob_y1
	# print prob_y1, prob_y0
	prob_y1_x = (probabilities[1]*prob_y1)/(probabilities[1]*prob_y1 + probabilities[0]*prob_y0)
	prob_y0_x = 1 - prob_y1_x
	print prob_y1_x, prob_y0_x
	if prob_y1_x > prob_y0_x:
		bestLabel = 1
		bestProb = prob_y1_x
	else:
		bestLabel = 0
		bestProb = prob_y0_x
	return bestLabel, bestProb



# 对整个测试集的类别进行预测
def getPreditions(summaries, testset, prob_y1):
	predictions = []
	probs = []
	for i in range(len(testset)):
		result, prob = predict2(summaries, testset[i], prob_y1)		# 这里没有写成这样testset[i,:-1]，是因为之后使用属性来求高斯概率遍历不到的尾部的标签
		predictions.append(result)
		probs.append(prob)
	return predictions, probs


# 计算精度
def getAccuracy(testset, predictions):
	correct = 0
	for i in range(len(testset)):
		if testset[i][-1] == predictions[i]:
			correct += 1
	accuracy = correct / float(len(testset)) * 100.0
	return accuracy



# 计算精度（precision）
# 精度是精确性的度量，表示被分为正例的示例中实际为正例的比例
def getPrecision(testset, predictions):
	true_positives = 0
	sums = 0
	for i in range(len(testset)):
		if predictions[i] == 1:
			sums += 1
			if testset[i][-1] == predictions[i]:
				true_positives = true_positives + 1
	precision = true_positives / float(sums) * 100.0
	return precision




# 计算召回率（recall）
# 召回率是覆盖面的度量，度量有多个正例被分为正例
def getRecall(testset, predictions):
	true_positives = 0
	sums = 0
	for i in range(len(testset)):
		if testset[i][-1] == 1:
			sums += 1
		if predictions[i] == 1 and testset[i][-1] == predictions[i]:
			true_positives = true_positives + 1
	recall = true_positives / float(sums) * 100.0
	return recall


def getF1(precision, recall):
	F1 = (2*precision*recall)/(precision + recall)
	return F1



def plotROC(predStrengths, classLabels):
	# print predStrengths
	cur = [1.0, 1.0]
	y_sum = 0.0
	nums_postives = np.sum(np.array(classLabels)==1)
	y_step = 1/float(nums_postives)
	x_step = 1/float(len(classLabels) - nums_postives)
	sorted_indicies = predStrengths.argsort()
	# print sorted_indicies
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	for index in sorted_indicies.tolist():
		if classLabels[index] == 1.0:
			del_x = 0
			del_y = y_step
		else:
			del_x = x_step
			del_y = 0
			y_sum += cur[1]
		ax.plot([cur[0], cur[0]-del_x], [cur[1], cur[1]-del_y], c='b')
		cur = (cur[0]-del_x, cur[1]-del_y)
	ax.plot([0,1],[0,1], 'b--')
	plt.title("power dataset classifyer's ROC with LSTM and Bayes")
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	ax.axis([0,1,0,1])
	plt.show()
	print 'the Area Under the Curve is:',y_sum*x_step



if __name__ == '__main__':
	# 获取误差数据集和loopsensor的数据集
	error_dataset, exception_label = createErrDataset()
	# for i in range(error_dataset.shape[1]-1):
	# 	print error_dataset[:,i]
	# 	plt.hist(error_dataset[:,i])
	# 	plt.show()
	prob_y1 = np.sum(exception_label)/float(len(exception_label))

	print error_dataset.shape
	# print power_data.shape

	acc_sum = 0.0
	epoch = 1
	split_ratio = 0.6
	for i in range(epoch):
		# 获取误差数据集大小的一个全排列
		# 作为索引以相同的方式打乱误差数据集和真实的车辆数据集
		index = np.random.permutation(len(error_dataset))
		error_dataset = error_dataset[index]
		# power_data = power_data[index]
		# 对误差数据集进行分离，分离出训练集和测试集
		trainset, testset = splitErrDataset(error_dataset, split_ratio)
		# 获取数据集中每个属性的信息（均值，标准差）	
		summaries = summarizeByClass(trainset)
		# predictions = getPreditions(summaries, testset)
		predictions, probs = getPreditions(summaries, testset, prob_y1)
		print 'truly:',testset[:,-1]
		print 'predict:',predictions
		accuracy = getAccuracy(testset, predictions)
		precision = getPrecision(testset, predictions)
		recall = getRecall(testset, predictions)
		F1 = getF1(precision, recall)
		print('Accuracy: {:.2f}%').format(accuracy)
		print('Precision: {:.2f}%').format(precision)
		print('Recall: {:.2f}%').format(recall)
		print('F1: {:.2f}%').format(F1)
		acc_sum = acc_sum + accuracy
		fpr, tpr, thresholds = roc_curve(testset[:,-1], np.array(probs))
		plotROC(np.array(probs), testset[:,-1])
	print thresholds
	# plt.plot([0, 1], [0, 1], 'k--')
	# plt.plot(fpr, tpr)
	# plt.show()
	
	print('Mean Accuracy: {:.2f}%').format(acc_sum/epoch)
	# errLocate = detectErrLocate(predictions, summaries, testset)
	# print errLocate
	# _, powerTest = splitErrDataset(powerDataset, 0.67)
