# coding: utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt

'''

本代码文件是利用电量数据来进行预测的，属于LSTM+Bayes的LSTM预测部分
代码总体思想是使用正常数据集来训练模型，然后测试集中夹杂着错误数据，来测试模型，接着利用预测数据和真实数据的误差来构建数据集合，然后使用高斯分布的贝叶斯来分类异常，并且侦查到异常的位置。
1.首先导入原电量数据集，处理成一周一周的，其中会舍去一年的前几天的数据，然后由于数据量大，进行下采样1/8
2.接着要把数据集进行归一化，即标准化，因为LSTM模型对数据很敏感，因此需要归一化
3.然后利用标签找出正常和异常的数据集，使用正常的数据集合进行训练。
4.再分别对训练集和测试集处理成模型的标准输入形状。
5.其中数据样式为，假设模型时间步timesteps=5
	输入： 1,2,3,4,5		标签：6
	输入： 2,3,4,5,6		标签：7
	输入： 3,4,5,6,7		标签：8
	... ... ... ...		...
6.构建LSTM模型，设定超参数，隐藏层的单元数可以和时间步设置相同的数量，同时在输出层加上一个全连接层。
7.在训练模型之前，注意选择正确的损失函数：
		多分类：tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y)
		二分类：tf.nn.sigmoid_cross_entropy_with_logits(pred_Y, Y)
8.注意预测的电量数据，还需要通过之前归一化之后得到的东西进行逆归一化，进行恢复。
9.分别计算训练集、测试集的预测数据与真实数据的误差，并存到本地，以待下个代码文件使用。

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
	print(downSample_data.shape)
	return downSample_data

'''
1.输入一周的数据，预测出下一周的数据
2.用正常的数据去训练
'''




def split_data(dataSet):
	time_cycles = dataSet.shape[1]
	# 把数据集分离成LSTM模型的输入，输出
	# 把0到50周的数据输入，1到51周的数据输入
	# 当然这些训练数据都是正常数据
	# data_x = dataSet[0:len(dataSet)-1]
	# data_y = dataSet[1:len(dataSet)]
	dataSet = dataSet.reshape(-1)
	# print dataSet.shape
	timesteps = 20
	data_x = [dataSet[i:i+timesteps] for i in xrange(len(dataSet)-timesteps-1+1)]
	data_x = np.array(data_x)
	data_y = [dataSet[i+timesteps] for i in xrange(len(dataSet)-timesteps)]
	data_y = np.array(data_y)
	# 把输入处理成[none, timesteps, features]形状的输入
	data_x = data_x.reshape([len(data_x), timesteps, 1])
	data_y = data_y.reshape([len(data_y), 1])
	return data_x, data_y


'''
LSTM对输入数据的比例敏感，特别是当使用Sigmoid（默认）或tanh激活函数时。 将数据重新调整到0到1的范围（也称为归一化）可能是一个很好的做法。 我们可以使用来自scikit-learn库的MinMaxScaler预处理类轻松地标准化数据集。

'''
def MinMaxScaling(data):	
	scaler = MinMaxScaler(feature_range=(0, 1))
	data_scaling = scaler.fit_transform(data)
	return data_scaling, scaler


# 将数据重新调整到0到1的范围（也称为归一化）
# http://www.cnblogs.com/chaosimple/p/4153167.html
def MinMaxNormalization(data, feature_range=(0,1)):
	norm = {}
	m = data.shape[0]
	MaxMin = np.zeros([m,2]);	Min = np.zeros([m,2])
	MaxMin[:,0] = np.array(range(51))
	Min[:,0] = np.array(range(51))


	MaxMin[:,1] = data.max(axis=1)-data.min(axis=1)
	Min[:,1] = data.min(axis=1)

	# norm['MaxMin'] = data.max(axis=1)-data.min(axis=1)
	# m = len(norm['MaxMin'])
	# norm['MaxMin'] = norm['MaxMin'].reshape([m,1])
	
	# norm['Min'] = data.min(axis=1)
	# norm['Min'] = norm['Min'].reshape([m,1])

	dataNormed = (data - Min[:,-1:])/MaxMin[:,-1:]
	dataNormed = dataNormed/(feature_range[1] - feature_range[0]) + feature_range[0]

	norm['MaxMin'] = MaxMin
	norm['Min'] = Min
	print(norm['Min'])
	return dataNormed, norm



def inverseNorm(data, norm, timesteps=20):
	data = data[84-20:]
	data = data.reshape([-1,84])
	data_inversed = data * norm['MaxMin']




def rnn_data(filename):
	dataSet = createDataSet(filename)
	# 下采样数据集
	dataSet = downSample(dataSet)
	# 平铺开以便于标准化
	# dataSet = dataSet.reshape([-1,1])
	# print dataSet.shape
	dataSet, norm = MinMaxNormalization(dataSet)

	# 归一化数据集
	# dataSet, scaler = MinMaxScaling(dataSet)  # (51,84)
	f = open('./power_label.txt','r')
	# 电量数据集正常与否的标签	
	label = []
	for line in f.readlines():
		line = line.strip()
		line = int(line)
		label.append(line)

	label = np.array(label)
	# dataSet = dataSet.reshape([51,84])
	normal_data = dataSet[label==1]

	train_norm_MaxMin = norm['MaxMin'][label==1]
	train_norm_Min = norm['Min'][label==1]
	train_norm = []
	train_norm.append(train_norm_MaxMin)
	train_norm.append(train_norm_Min)
	print(len(normal_data))
	
	train_x, train_y = split_data(normal_data)
	abnormal_data = dataSet[label==0]

	test_norm = []
	
	test_norm_MaxMin = norm['MaxMin'][label==0]
	test_norm_Min = norm['Min'][label==0]
	
	test_norm.append(test_norm_MaxMin)
	test_norm.append(test_norm_Min)


	test_x, test_y = split_data(abnormal_data)

	return train_x, train_y, test_x, test_y, train_norm, test_norm


	

class Config(object):
	def __init__(self, train_x):
		self.timesteps = train_x.shape[1]
		self.features = train_x.shape[2]
		self.outputdims = train_y.shape[1]

		self.learning_rate = 0.015
		self.lambda_loss_amount = 0.015
		self.training_epoch = 1000
		self.minibatch_size = 100
		self.keep_prob = 0.5

		self.hidden_nums = 6
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



def calculateMSE(trueValue, predictValue, scaler):
	predictValue = scaler.inverse_transform(predictValue)
	print(predictValue.shape)
	sums = np.sum(np.square(trueValue - predictValue))
	# MSE = np.mean(np.sum(np.square(trueValue - predictValue))) BIG ERROR
	MSE = np.mean(np.square(trueValue - predictValue))
	print(sums, MSE)
	return MSE

def calculateMSE2(trueValue, predictValue):
	print(predictValue.shape)
	sums = np.sum(np.square(trueValue - predictValue))
	# MSE = np.mean(np.sum(np.square(trueValue - predictValue))) BIG ERROR
	MSE = np.mean(np.square(trueValue - predictValue))
	print(sums, MSE)
	return MSE



def calculateMSE3(trueValue, predictValue, norm):
	predictValue = norm.inverse_transform(predictValue)
	print(predictValue.shape)
	sums = np.sum(np.square(trueValue - predictValue))
	# MSE = np.mean(np.sum(np.square(trueValue - predictValue))) BIG ERROR
	MSE = np.mean(np.square(trueValue - predictValue))
	print(sums, MSE)
	return MSE




if __name__ == '__main__':
	train_x, train_y, test_x, test_y, norm = rnn_data('./power_data.txt')
	print(train_x.shape)
	print(train_y.shape)
	# test_x = train_x[0]
	# test_y = train_y[0]
	print(test_y.shape)
	print(test_x.shape)
	
	# sh = test_x.shape
	# sh = (1,) + sh
	# test_x = test_x.reshape(sh)
	# test_y = test_y.reshape([1, len(test_y)])


	# Building Graphs
	config = Config(train_x)

	X = tf.placeholder(tf.float32, [None, config.timesteps, config.features])
	Y = tf.placeholder(tf.float32, [None, config.outputdims])

	# setting parameter
	epoch = config.training_epoch
	lr = config.learning_rate
	batch_size = config.minibatch_size

	# forward calcs
	# pred_Y = LSTM_Network(train_x, config)	TypeError: Tensors in list passed to 'values' of 'Concat' Op have types [float64, float32] that don't all match.
	pred_Y = LSTM_Network(X, config)

	# cost functions
	# 要用平均平方误差，不能用总平方误差
	# cost = tf.reduce_sum(tf.square(pred_Y - Y)) ————错误的
	cost = tf.reduce_mean(tf.square(pred_Y - Y))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	
	# 生成saver
	saver = tf.train.Saver()

	# Now training the model	
	# start session
	sess = tf.Session()
	# 初始化所有变量
	init = tf.initialize_all_variables()
	sess.run(init)

	# 
	test_losses = []
	train_losses = []

	# 



	# iterative training
	# epoch training
	for i in range(epoch):
		for start, end in zip(range(0, len(train_x), batch_size), 
							  range(batch_size, len(train_x)+1, batch_size)):
			sess.run(optimizer, feed_dict = {X: train_x[start:end],
											 Y: train_y[start:end]})		# 分离出batchsize个数据去迭代运算

		# cost = sess.run(cost, feed_dict = {X: test_x, Y: test_y})		# 之前犯错的地方，就是因为cost操作被再次赋值，导致出错！
		# loss, train_result = sess.run([cost, pred_Y], feed_dict = {X: train_x, Y: train_y})
		loss_train = sess.run(cost, feed_dict = {X: train_x, Y: train_y})
		# loss_train = calculateMSE2(train_y, train_result)


		loss_test = sess.run(cost, feed_dict = {X: test_x, Y: test_y})
		# loss_test = calculateMSE2(test_y, test_result)
				
		train_losses.append(loss_train)		
		test_losses.append(loss_test)		
		
		print('the epoch'+str(i+1)+': train loss = '+'{:.9f}'.format(loss_train))

	test_result = sess.run(pred_Y, feed_dict={X:test_x, Y:test_y})
	# test_result = scaler.inverse_transform(test_result)				# 返回归一化之前的模样
	train_result = sess.run(pred_Y, feed_dict={X:train_x, Y:train_y})
	# train_result = scaler.inverse_transform(train_result)	# 返回归一化之前的模样
	# np.savetxt('test_result7.txt',test_result)
	# np.savetxt('train_result7.txt',train_result)
	# saver.save(sess, './power/model')
	# np.savetxt('train_losses.txt',train)


	'''
	#Training is good, but having visual insight is even better

	'''
	font = {
		'family' : 'Bitstream Vera Sans',
		'weight' : 'bold',
		'size'   : 18
	}
	matplotlib.rc('font', **font)

	width = 12
	height = 12
	plt.figure(figsize=(width, height))

	# indep_train_axis = np.array(range(config.batch_size, (len(train_losses)+1)*config.batch_size, config.batch_size))
	plt.plot(range(1,epoch+1), np.array(train_losses),     "b--", label="Train losses")
	# plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

	# indep_test_axis = np.array(range(config.batch_size, len(test_losses)*config.display_iter, config.display_iter)[:-1] + [config.train_count*config.training_epochs])
	plt.plot(range(1,epoch+1), np.array(test_losses),     "b-", label="Test losses")
	# plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

	plt.title("Training session's progress over iterations")
	plt.legend(loc='upper right', shadow=True)
	plt.ylabel('Training Progress (Loss or Accuracy values)')
	plt.xlabel('Training iteration')

	plt.show()
