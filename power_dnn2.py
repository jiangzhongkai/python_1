# coding: utf-8

'''
这个代码文件为powe_demand2.py的对比实验
该文件是利用深度神经网络的方法来侦测电量的异常

'''

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random

# 
def createDataSet(filename):
	f = open(filename,'rb')
	data = []
	for line in f.readlines():
		line = int(line.strip())
		data.append(line)
	
	data = data[480:]		# 480, 因为1997年的前五天是从周三开始算的，故选择舍弃掉前5天的数据，按照1天96个数据，96*5=480
	dataSet = []

	for i in range(0, len(data)-672, 672):	# one hour is 4 point, then a day is need 24*4=96 point, so a week is 96*7=672 point
		sequence = data[i:i+672]
		dataSet.append(sequence)

	dataSet = np.array(dataSet)
	return dataSet

# 下采样
# 将时间序列中每８个值用１个值代替，这里选择８个值的均值
# 之前是15分钟一个值，一天96个值
# 现在降采样后，2小时一个值，一天12个值
def downSample(dataSet):
	downSample_data = []
	for i in range(len(dataSet)):
		line = np.ones([672/8])
		j = 0
		for start, end in zip(range(0,672,8), range(8,673,8)):
			temp_array = dataSet[i,start:end]
			temp_array = np.mean(temp_array)
			line[j] = temp_array
			j = j + 1
		downSample_data.append(line)
	downSample_data = np.array(downSample_data)
	return downSample_data



def extendeDataset(dataSet):
	extendeData = []
	for m in range(len(dataSet)):
		size = 672/8	# 计算分离成8条时间序列，每条时间序列的长度
		# 创建一个形状为8X84的数据，即存储着8条84长度的时间序列		
		data = np.zeros([8,size])
		for i in range(8):
			index = i
			for j in range(size):
				data[i,j] = dataSet[m,index]
				index = index + 8	# 把原始数据集角标往后移动8位
		extendeData.extend(data)	# 使用extend函数而不是append函数，使得其直接成为二维数组
	print np.array(extendeData).shape #（408,84）,408=51*8
	return np.array(extendeData)




def split_data(dataSet, splitRatio=0.75):
	trainingNums = int(len(dataSet)*splitRatio)
	trainset = dataSet[:trainingNums]
	testset = dataSet[trainingNums:]
	
	# 内部函数，便于重复调用，分离输入数据和标签
	def _separate_data(dataSet):
		data_x = dataSet[:,:-1]
		data_y = dataSet[:,-1]
		return data_x, data_y
	
	train_x, train_y = _separate_data(trainset)
	test_x, test_y = _separate_data(testset)
	return train_x, train_y, test_x, test_y

'''
LSTM对输入数据的比例敏感，特别是当使用Sigmoid（默认）或tanh激活函数时。 将数据重新调整到0到1的范围（也称为归一化）可能是一个很好的做法。 我们可以使用来自scikit-learn库的MinMaxScaler预处理类轻松地标准化数据集。

'''
def MinMaxScaling(data):	
	scaler = MinMaxScaler(feature_range=(0, 1))
	data_scaling = scaler.fit_transform(data)
	return data_scaling, scaler



def MinMaxNormalization(data, feature_range=(0,1)):
	norm = {}
	m = data.shape[0]
	# MaxMin = np.zeros([m,1]);	Min = np.zeros([m,1])
	# MaxMin[:,0] = np.array(range(m))
	# Min[:,0] = np.array(range(m))


	MaxMin = data.max(axis=1)-data.min(axis=1)
	Min = data.min(axis=1)		# 求data中每一行的最小值，axis=1代表把每一列来进行计算，即求出由最小值组成的一列

	MaxMin = MaxMin.reshape([-1,1])
	Min = Min.reshape([-1,1])
	
	dataNormed = (data - Min)/MaxMin
	dataNormed = dataNormed/(feature_range[1] - feature_range[0]) + feature_range[0]

	norm['MaxMin'] = MaxMin
	norm['Min'] = Min
	# print norm['Min']
	return dataNormed, norm



def dnn_data(filename, splitRatio=0.75):
	dataSet = createDataSet(filename)
	# 扩展数据集
	extend_dataSet = extendeDataset(dataSet)
	# 下采样数据集
	dataSet = downSample(dataSet)
	f = open('./power_label.txt','r')
	# 电量数据集正常与否的标签	
	label = []
	for line in f.readlines():
		line = line.strip()
		line = int(line)
		label.append(line)
	label = np.array(label)
	# 获取扩展数据集的标签
	extend_data_label = np.array(label).repeat(8)
	normal_data = dataSet[label==1]
	# trainset, train_norm = MinMaxNormalization(normal_data)
	# train_x, train_y = split_data(trainset)
	# 获取异常数据
	abnormal_data = dataSet[label==0]
	# 扩展异常数据集
	extend_abnormal_data = extend_dataSet[extend_data_label==0]
	absample_index = random.sample(range(len(extend_abnormal_data)), 30)
	extend_abnormal_data = extend_abnormal_data[absample_index]
	# 连接本来的异常数据和扩展的异常数据	
	abnormal_data = np.concatenate((abnormal_data, extend_abnormal_data), axis=0)
	dataset = np.concatenate((normal_data, abnormal_data), axis=0)
	# dataset, norm = MinMaxNormalization(dataset)
	dataset, _ = MinMaxScaling(dataset)
	# 是否为异常的标签，1代表是异常
	exception_label = [0 for i in range(len(normal_data))] + [1 for i in range(len(abnormal_data))]
	exception_label = np.array(exception_label).reshape([len(exception_label), 1])
	# 将纯误差数据和是否异常的标签合并
	dataset = np.concatenate((dataset, exception_label), axis=1) 
	
	np.random.shuffle(dataset)

	return split_data(dataset, splitRatio=0.75)



# 计算accuracy
def getAccuracy(true, predictions):
	correct = 0
	for i in range(len(true)):
		if true[i] == predictions[i]:
			correct += 1
	accuracy = correct / float(len(true)) * 100.0
	return accuracy


# 计算精度（precision）
# 精度是精确性的度量，表示被分为正例的示例中实际为正例的比例
def getPrecision(testset, predictions):
	true_positives = 0
	sums = 0
	for i in range(len(testset)):
		if predictions[i] == 1:
			sums += 1
			if testset[i] == predictions[i]:
				true_positives = true_positives + 1
	if sums == 0:
		return 0
	precision = true_positives / float(sums) * 100.0
	return precision




# 计算召回率（recall）
# 召回率是覆盖面的度量，度量有多个正例被分为正例
def getRecall(testset, predictions):
	true_positives = 0
	sums = 0
	for i in range(len(testset)):
		if testset[i] == 1:
			sums += 1
		if predictions[i] == 1 and testset[i] == predictions[i]:
			true_positives = true_positives + 1
	recall = true_positives / float(sums) * 100.0
	return recall


def getF1(precision, recall):
	if (precision + recall) == 0:
		return 0
	else:
		F1 = (2*precision*recall)/(precision + recall)
		return F1




if __name__ == '__main__':
	accuracy = 0.0
	for i in xrange(10):
		# load dataset
		train_x, train_y, test_x, test_y = dnn_data('./power_data.txt')
		
		# train model
		classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10,20,10],n_classes=2)
		classifier.fit(x=train_x,y=train_y,steps=1000)

		#预估模型的正确率 evaluate()方法返回的是一个字典
		accuracy_score = classifier.evaluate(x=test_x, y=test_y)["accuracy"]
		print("Accuracy: {:.2f}%".format(accuracy_score*100))
		accuracy = accuracy + accuracy_score

	mean_accuracy = float(accuracy)/10
	print("Accuracy: {:.2f}%".format(mean_accuracy*100))

	# Classify two new flower samples.
	
	new_samples = test_x
	y = classifier.predict(new_samples)
	y = y.reshape([-1,1])
	precision = getPrecision(test_y, y)
	acc = getAccuracy(test_y, y)
	recall = getRecall(test_y, y)
	f1 = getF1(precision, recall)
	print('Precision: {:.2f}%').format(precision)
	print('Accuracy: {:.2f}%').format(acc)
	print('Recall: {:.2f}%').format(recall)
	print('F1: {:.2f}%').format(f1)
