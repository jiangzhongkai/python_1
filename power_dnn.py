# coding: utf-8

'''
这个代码文件为powe_demand2.py的对比实验
该文件是利用深度神经网络的方法来侦测电量的异常

'''

import tensorflow as tf
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Data sets
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



'''
扩展数据集
通过将一条长度为672的电量时间序列，分离成8条长度为84的时间序列
这与之前的下采样是不同的。主要思想如下：
1.首先计算分离成8条时间序列，每条时间序列的长度
2.针对每一条672长度的时间序列，将每8个值，进行分离，第一条序列得第1个值，第二条序列得第2个值，
	以此类推，第八条序列得第8个值。
由此就扩充了数据集，而不是单独得取平均值，舍弃掉了一些信息。

'''
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
	# print np.array(extendeData).shape #（408,84）,408=51*8
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




def dnn_data(filename, splitRatio=0.75):
	dataSet = createDataSet(filename)
	# 扩展数据集
	dataSet = extendeDataset(dataSet)
	# 平铺开以便于标准化
	# dataSet = dataSet.reshape([-1,1])
	# print dataSet.shape
	# dataSet, norm = MinMaxNormalization(dataSet)

	f = open('./power_label.txt','r')
	# 电量数据集正常与否的标签	
	label = []
	for line in f.readlines():
		line = line.strip()
		line = int(line)
		label.append(line)

	newLabel = np.array(label).repeat(8)	# 一定要加上，之前忘记加上去了，将标签每个值复制8次
	newLabel = newLabel[:,np.newaxis]
	print newLabel.shape
	print dataSet.shape
	newDataSet = np.concatenate([dataSet, newLabel], axis=1)
	
	np.random.shuffle(newDataSet)

	return split_data(newDataSet, splitRatio=0.75)



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
	for i in xrange(5):
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
