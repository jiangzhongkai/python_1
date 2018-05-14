# coding: utf-8
import numpy as np
import kmeans_dtw as kt
import tensorflow as tf

# def createDataSet(filename):
# 	f = open(filename,'rb')
# 	data = []
# 	for line in f.readlines():
# 		line = int(line.strip())
# 		data.append(line)
# 	# nums= len(data)/750.0 	# 46.72

 
# 	data = data[480:]		# 480, 因为1997年的前五天是从周三开始算的，故选择舍弃掉前5天的数据，按照1天96个数据，96*5=480
# 	dataSet = []

# 	for i in range(0, len(data)-672, 672):	# one hour is 4 point, then a day is need 24*4=96 point, so a week is 96*7=672 point
# 		sequence = data[i:i+672]
# 		dataSet.append(sequence)

# 	dataSet = np.array(dataSet)
# 	return dataSet

def split_data(dataSet):
	time_steps = dataSet.shape[1]
	# 把数据集分离成LSTM模型的输入，输出
	# 以1到83个时间步为输入，2到84个时间步为输出
	data_x = dataSet[:, :time_steps-1]
	data_y = dataSet[:, 1:]
	# sh = data_x.shape
	# sh = sh + (1,)
	# data_x = data_x.reshape(sh)
	# 把输入处理成[none, timesteps, features]形状的输入
	data_x = data_x.reshape([len(data_x), time_steps-1, 1])
	return data_x, data_y



def rnn_data(filename):
	dataSet = kt.createDataSet(filename)
	# 下采样数据集
	dataSet = kt.downSample(dataSet)
	f = open('./power_label.txt','r')
	# 电量数据集正常与否的标签	
	label = []
	for line in f.readlines():
		line = line.strip()
		line = int(line)
		label.append(line)

	label = np.array(label)
	normal_data = dataSet[label==1]
	print(len(normal_data))
	# time_steps = normal_data.shape[1]
	# data_x = normal_data[:, :time_steps-1]
	# data_y = normal_data[:, 1:]
	# sh = data_x.shape
	# sh = sh + (1,)
	# data_x = data_x.reshape(sh)
	train_x, train_y = split_data(normal_data)
	abnormal_data = dataSet[label==0]
	test_x, test_y = split_data(abnormal_data)

	return train_x, train_y, test_x, test_y


	

class Config(object):
	def __init__(self, train_x):
		self.timesteps = train_x.shape[1]
		self.features = train_x.shape[2]

		self.learning_rate = 0.1
		self.lambda_loss_amount = 0.015
		self.training_epoch = 1000
		self.minibatch_size = 2

		self.hidden_one = 1
		self.hidden_two = 1
		self.W = {
			'output': tf.Variable(tf.random_normal([self.hidden_two, 83]))
		}
		self.biases = {
			'output': tf.Variable(tf.random_normal([83]))
		}


def LSTM_Network(input_data, config):
	input_data = tf.transpose(input_data, [1, 0, 2])
	input_data = tf.reshape(input_data, [-1, config.features])

	# New input_data's shape: a list of lenght "timesteps" containing tensors of shape [batch_size, features]
	input_data = tf.split(0, config.timesteps, input_data)
	# 处理成一个一个时间步的输入到模型的数据
	
	lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_one, forget_bias=1.0, state_is_tuple=True)
	lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_two, forget_bias=1.0, state_is_tuple=True)

	stack_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1, lstm_cell2], state_is_tuple=True)

	outputs, _ = tf.nn.rnn(stack_lstm, input_data, dtype=tf.float32)

	# output = tf.concat(1, outputs)

	output = tf.matmul(outputs[-1], config.W['output']) + config.biases['output']

	return output



if __name__ == '__main__':
	train_x, train_y, test_x, test_y = rnn_data('./power_data.txt')
	# test_x = train_x[0]
	# test_y = train_y[0]
	# print test_y.shape
	# print test_x.shape
	
	# sh = test_x.shape
	# sh = (1,) + sh
	# test_x = test_x.reshape(sh)
	# test_y = test_y.reshape([1, len(test_y)])


	# Building Graphs
	config = Config(train_x)

	X = tf.placeholder(tf.float32, [None, config.timesteps, config.features])
	Y = tf.placeholder(tf.float32, [None, config.timesteps])

	# setting parameter
	epoch = config.training_epoch
	lr = config.learning_rate
	batch_size = config.minibatch_size

	# forward calcs
	# pred_Y = LSTM_Network(train_x, config)	TypeError: Tensors in list passed to 'values' of 'Concat' Op have types [float64, float32] that don't all match.
	pred_Y = LSTM_Network(X, config)

	# cost functions
	cost = tf.reduce_sum(tf.square(pred_Y - Y))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	
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
		loss = sess.run(cost, feed_dict = {X: train_x, Y: train_y})		
		# loss = sess.run(cost, feed_dict = {X: test_x, Y: test_y})		
		print('the epoch'+str(i+1)+': loss = '+'{:.6f}'.format(loss))

	test_result = sess.run(pred_Y, feed_dict={X:test_x, Y:test_y})
	train_result = sess.run(pred_Y, feed_dict={X:train_x, Y:train_y})
	np.savetxt('test_result4.txt',test_result)
	np.savetxt('train_result4.txt',train_result)

	# predict_train = sess.run(pred_Y, feed_dict = {X: train_x, Y: train_y})
	# predict_test = sess.run(pred_Y, feed_dict = {X: test_x, Y: test_y})

	# labels = np.concatenate((train_y, test_y))		# 数组拼接
	# predict_train = scaler.inverse_transform(predict_train)		# 返回归一化之前的模样
	# predict_test = scaler.inverse_transform(predict_test)
	# labels = scaler.inverse_transform(labels)

	# plt.plot(labels, label='true values')
	# plt.plot(predict_train, label = 'predicted train')
	# plt.plot(predict_test, label = 'predicted test')

	# plt.legend(loc = 'upper right')
	# plt.show()
