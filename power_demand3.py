# coding: utf-8
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt




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
	print np.array(extendeData).shape #（408,84）,408=51*8
	return np.array(extendeData)



# 根据每个时间步窗口的数据进行归一化
# 参考于https://yq.aliyun.com/articles/68463
def normalise_windows(window_data):
	normalised_data = []
	inverse_factor = []
	for window in window_data:
		normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
		normalised_data.append(normalised_window)
		inverse_factor.append(window[0])
	return normalised_data, inverse_factor


def split_data(dataSet):
	time_cycles = dataSet.shape[1]
	dataSet = dataSet.reshape(-1)
	# print dataSet.shape
	timesteps = 12
	data = [dataSet[i:i+timesteps+1] for i in xrange(len(dataSet)-timesteps-1)]
	# data, inverse_factor = normalise_windows(data)
	data_x = np.array(data)[:, :-1]
	data_y = np.array(data)[:, -1]
	# data_y = [dataSet[i+timesteps] for i in xrange(len(dataSet)-timesteps)]
	# data_y = np.array(data_y)
	# 把输入处理成[none, timesteps, features]形状的输入
	data_x = data_x.reshape([len(data_x), timesteps, 1])
	data_y = data_y.reshape([len(data_y), 1])
	return data_x, data_y





def rnn_data(filename, splitRatio=0.9):
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

	label = np.array(label)
	newLabel = np.array(label).repeat(8)	# 一定要加上，之前忘记加上去了，将标签每个值复制8次

	normal_data = dataSet[newLabel==1]
	print 'normal_data:',len(normal_data)
	abnormal_data = dataSet[newLabel==0]
	
	trainingNums = int(len(normal_data)*splitRatio)
	trainset = normal_data[:trainingNums]
	otherset = normal_data[trainingNums:]
	testset = np.concatenate((otherset, abnormal_data), axis=0)
	print len(trainset)
	print len(testset)
	
	train_x, train_y = split_data(trainset)
	
	test_x, test_y = split_data(testset)

	return train_x, train_y, test_x, test_y



def calculateMSE(trueValue, predictValue, inverse_factor):
	m,n = predictValue.shape
	predictInversed = np.zeros([m,n])
	for i in range(m):
		predictInversed[i] = (predictValue[i] + 1) * inverse_factor[i]

	print predictInversed.shape
	# MSE = np.mean(np.sum(np.square(trueValue - predictValue))) BIG ERROR
	MSE = np.mean(np.square(trueValue - predictInversed))
	print MSE
	return MSE




class Config(object):
	def __init__(self, train_x):
		self.timesteps = train_x.shape[1]
		self.features = train_x.shape[2]
		self.outputdims = train_y.shape[1]

		self.learning_rate = 0.015
		self.lambda_loss_amount = 0.015
		self.training_epoch = 500
		self.minibatch_size = 100
		self.keep_prob = 0.5

		self.hidden_nums = 20
		# self.hidden_two = 168
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

	# saver.restore(sess, './power2/model')

	# iterative training
	# epoch training
	for i in range(epoch):
		for start, end in zip(range(0, len(train_x), batch_size), 
							  range(batch_size, len(train_x)+1, batch_size)):
			sess.run(optimizer, feed_dict = {X: train_x[start:end],
											 Y: train_y[start:end]})		# 分离出batchsize个数据去迭代运算

		# cost = sess.run(cost, feed_dict = {X: test_x, Y: test_y})		# 之前犯错的地方，就是因为cost操作被再次赋值，导致出错！
		# _, train_result = sess.run([cost, pred_Y], feed_dict = {X: train_x, Y: train_y})
		loss_train = sess.run(cost, feed_dict = {X: train_x, Y: train_y})
		# loss_train = calculateMSE(train_y, train_result, inverse_factor_train)


		# _, test_result = sess.run([cost, pred_Y], feed_dict = {X: test_x, Y: test_y})
		loss_test = sess.run(cost, feed_dict = {X: test_x, Y: test_y})
		# loss_test = calculateMSE(test_y, test_result, inverse_factor_test)
				
		train_losses.append(loss_train)		
		test_losses.append(loss_test)		
		
		print 'the epoch'+str(i+1)+': train loss = '+'{:.3f}'.format(loss_train)

	test_result = sess.run(pred_Y, feed_dict={X:test_x, Y:test_y})
	# test_result = scaler.inverse_transform(test_result)				# 返回归一化之前的模样
	train_result = sess.run(pred_Y, feed_dict={X:train_x, Y:train_y})
	# train_result = scaler.inverse_transform(train_result)	# 返回归一化之前的模样
	# np.savetxt('test_result7.txt',test_result)
	# np.savetxt('train_result7.txt',train_result)
	saver.save(sess, './power2/model')
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
