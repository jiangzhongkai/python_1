# coding: utf-8
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import random

'''

这个代码文件为powe_demand2.py的对比实验
本代码是思想是仅仅采用LSTM算法来进行分类
首先把数据集进行混洗，分割出训练集和测试集
然后使用一周的电量数据作为输入，正常与否（1或0）作为标签
导入模型中，构建模型，再训练模型，设定超参数
接着打印正确率

'''


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
	timesteps = dataSet.shape[1] - 1
	# 把数据集分离成LSTM模型的输入，输出
	trainingNums = int(len(dataSet) * splitRatio)
	trainset = dataSet[:trainingNums]
	testset = dataSet[trainingNums:]
	
	# 内部函数，便于重复调用，分离输入数据和标签
	def _separate_data(dataSet):
		data_x = dataSet[:,:-1]
		# 把输入处理成[none, timesteps, features]形状的输入
		data_x = data_x.reshape([len(data_x), timesteps, 1])
		data_y = dataSet[:,-1:]		# 特意是[:,-1:],而不是[:,-1]是为了切片时不影响原数组的维数，保持原有二维
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



def rnn_data(filename, splitRatio=0.75):
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




class Config(object):
	def __init__(self, train_x):
		self.timesteps = train_x.shape[1]
		self.features = train_x.shape[2]
		self.outputdims = train_y.shape[1]

		self.learning_rate = 0.015
		self.lambda_loss_amount = 0.015
		self.training_epoch = 1000
		self.minibatch_size = 10
		self.keep_prob = 0.5

		self.hidden_nums = 20
		# self.hidden_two = 84
		self.W = {
			'hidden': tf.Variable(tf.random_normal([self.features, self.hidden_nums])),
			'output': tf.Variable(tf.random_normal([self.hidden_nums, 1]))
		}
		self.biases = {
			'hidden': tf.Variable(tf.random_normal([self.hidden_nums])),
			'output': tf.Variable(tf.random_normal([1]))
		}


def LSTM_Network(input_data, config):
	input_data = tf.transpose(input_data, [1, 0, 2])
	input_data = tf.reshape(input_data, [-1, config.features])

	# input_data = tf.matmul(
 #        input_data, config.W['hidden']
 #    ) + config.biases['hidden']

	# New input_data's shape: a list of lenght "timesteps" containing tensors of shape [batch_size, hidden_one]
	input_data = tf.split(0, config.timesteps, input_data)
	# 处理成一个一个时间步的输入到模型的数据

	lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_nums, forget_bias=1.0, state_is_tuple=True)
	# dropout lstm_cell2
	lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_nums, forget_bias=1.0, state_is_tuple=True)
	# lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell2, config.keep_prob)
	# stack lstm cell
	stack_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1, lstm_cell2], state_is_tuple=True)

	outputs, _ = tf.nn.rnn(stack_lstm, input_data, dtype=tf.float32)

	# output = tf.concat(1, outputs)	# 这个函数在最新的版本中已经更新函数的参数列表了

	output = tf.matmul(outputs[-1], config.W['output']) + config.biases['output']

	return output


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
	train_x, train_y, test_x, test_y = rnn_data('./power_data.txt')
	print train_x.shape
	print train_y.shape
	
	print test_x.shape
	print test_y.shape
	
	


	# Building Graphs
	config = Config(train_x)

	X = tf.placeholder(tf.float32, [None, config.timesteps, config.features])
	Y = tf.placeholder(tf.float32, [None, config.outputdims])

	# setting parameter
	epoch = config.training_epoch
	lr = config.learning_rate
	batch_size = config.minibatch_size

	# forward calcs
	# pred_y = LSTM_Network(train_x, config)	TypeError: Tensors in list passed to 'values' of 'Concat' Op have types [float64, float32] that don't all match.
	pred_y = LSTM_Network(X, config)

	# cost functions
	cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pred_y, Y))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	
	# 计算正确率，由于是二分类，所以使用sigmod函数
	# tf.round()函数为四舍五入函数，sig结果大于0.5便为1，小于0.5便为0，以此得到分类结果
	# 然后再与实际结果相比较，使用tf.equal函数得到[[ True],[ True],[ True],[ True],[ True],[False],[ True],[False],....,[ True],[ True],[False],[ True]], shape=[38,1]

	sig = tf.sigmoid(pred_y)
	rd = tf.round(tf.sigmoid(pred_y))
	correct_nums = tf.equal(tf.round(tf.sigmoid(pred_y)), Y)

	# tf.cast() Casts a tensor to a new type.
	# tf.reduce_mean() 求平均数
	accuracy = tf.reduce_mean(tf.cast(correct_nums, dtype=tf.float32))
	
	# Now training the model	
	# start session
	sess = tf.Session()
	# 初始化所有变量
	init = tf.initialize_all_variables()
	sess.run(init)
	# iterative training
	# epoch training
	for i in range(epoch):
		for start, end in zip(range(0, len(train_x), batch_size), 
							  range(batch_size, len(train_x)+1, batch_size)):
			sess.run(optimizer, feed_dict = {X: train_x[start:end],
											 Y: train_y[start:end]})		# 分离出batchsize个数据去迭代运算

		# cost = sess.run(cost, feed_dict = {X: test_x, Y: test_y})		# 之前犯错的地方，就是因为cost操作被再次赋值，导致出错！
		trainingAcc, train_correctnums, predict_train, sig_train = sess.run([accuracy, correct_nums, sig, rd], feed_dict = {X: train_x, Y: train_y})
		print train_correctnums.shape		# shape=[38,1] 代表38个训练样本

		testingAcc, predict_test = sess.run([accuracy,rd], feed_dict = {X: test_x, Y: test_y})		
		# loss = sess.run(cost, feed_dict = {X: test_x, Y: test_y})		
		# format()方法打印百分比，详情请看http://www.th7.cn/Program/Python/201604/843186.shtml
		#  print('{0:.2%}'.format(0.34)) #打印百分数，指定保留2位小数
		print 'Epoch '+str(i+1)+\
			  ', training accuracy: {0:.2%}'.format(trainingAcc)+\
			  ', test accuracy: {0:.2%}'.format(testingAcc)

		# print predict_train, sig_train
		precis_train = getPrecision(train_y, predict_train)
		
		precis_test = getPrecision(test_y, predict_test)
		acc_test = getAccuracy(test_y, predict_test)

		recall_test = getRecall(test_y, predict_test)

		f1 = getF1(precis_test, recall_test)


		print('Precision: {:.2f}%').format(precis_test)
		print('Accuracy: {:.2f}%').format(acc_test)
		print('Recall: {:.2f}%').format(recall_test)
		print('F1: {:.2f}%').format(f1)

