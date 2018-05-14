# coding: utf-8
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt

'''

在power_demand4的基础上把下采样数据替换成扩展数据集了




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


'''
构建一个shape为[none,672]的数组，意思是一周672个数据，none的意思是这个数据集包含多周数据
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


'''下采样
1.将时间序列中每8个值用1个值代替，这里选择8个值的均值
2.之前是15分钟一个值，一天96个值，另672代表一周的数据 96*7=672
3.现在降采样后，2小时一个值，一天12个值
'''
def downSample(dataSet):
	downSample_data = []
	# 遍历数据集中每一周的数据，然后进行每8个值用1个值代替
	for i in range(len(dataSet)):
		line = np.ones([672/8])	# 创建一个数组来装替换后一周的数据，故672/8
		j = 0
		# 使用zip操作构建诸如[0,8],[8,16],...[664,672]这样的区间角标
		for start, end in zip(range(0,672,8), range(8,673,8)):
			temp_array = dataSet[i,start:end]
			temp_array = np.mean(temp_array)
			line[j] = temp_array
			j = j + 1
		downSample_data.append(line)
	downSample_data = np.array(downSample_data)
	print downSample_data.shape
	return downSample_data



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



def split_data(dataSet):
	time_cycles = dataSet.shape[1]
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

'''
# 将数据重新调整到0到1的范围（也称为归一化）
# http://www.cnblogs.com/chaosimple/p/4153167.html
# 我们根据时间序列的特性，采用行归一化，即把每一条时间序列归一化到[0,1]的区间内
'''
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



def rnn_data(filename):
	dataSet = createDataSet(filename)
	# 扩展数据集
	dataSet = extendeDataset(dataSet)
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
	# 有train_norm和test_norm之分是因为两个数据集是分开归一化的，但是这并不会影响模型的输入，
	# 因为我们不是对列归一化，而是行归一化，就算两个数据集一起归一化，这与数据集分开归一化没什么区别。
	# 若是进行列归一化，则需要在数据集分割之前进行归一化
	trainset, train_norm = MinMaxNormalization(normal_data)
	train_x, train_y = split_data(trainset)
	# 获取异常数据
	abnormal_data = dataSet[newLabel==0]
	testset, test_norm = MinMaxNormalization(abnormal_data)
	test_x, test_y = split_data(testset)
	
	return train_x, train_y, test_x, test_y, train_norm, test_norm


	

class Config(object):
	def __init__(self, train_x):
		self.timesteps = train_x.shape[1]
		self.features = train_x.shape[2]
		self.outputdims = train_y.shape[1]

		self.learning_rate = 0.0075
		self.lambda_loss_amount = 0.015
		self.training_epoch = 1000
		self.minibatch_size = 500
		self.keep_prob = 0.5

		self.hidden_nums = 10
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




# 图像比较
def plot_compare(dataset1, dataset2):
	for i in range(len(dataset1)):
		plt.plot(dataset1[i], label='true')
		plt.plot(dataset2[i], label='predict')
		plt.legend(loc = 'upper right')
		plt.show()



def calculateMSE(trueValue, predictValue, norm):
	# 把预测的结果进行数据还原
	predictValue = inverseNorm(predictValue, norm)
	print predictValue.shape
	# 真实的结果也需要还原，因为之前真实的数据也进行了归一化
	trueValue = inverseNorm(trueValue, norm)
	# MSE = np.mean(np.sum(np.square(trueValue - predictValue))) BIG ERROR
	MSE = np.mean(np.square(trueValue - predictValue))
	print MSE
	return MSE, trueValue, predictValue




if __name__ == '__main__':
	train_x, train_y, test_x, test_y, train_norm, test_norm = rnn_data('./power_data.txt')
	print train_x.shape
	print train_y.shape
	print test_y.shape
	print test_x.shape

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
		loss = sess.run(cost, feed_dict = {X: train_x, Y: train_y})
		
		train_result = sess.run(pred_Y, feed_dict = {X: train_x, Y: train_y})
		loss_train, _, _ = calculateMSE(train_y, train_result, train_norm)


		test_result = sess.run(pred_Y, feed_dict = {X: test_x, Y: test_y})
		loss_test, _, _ = calculateMSE(test_y, test_result, test_norm)
				
		train_losses.append(loss_train)		
		test_losses.append(loss_test)		
		
		print 'the epoch'+str(i+1)+': train loss = '+'{:.9f}'.format(loss)


	MSE, trueValue, predictValue = calculateMSE(train_y, train_result, train_norm)
	plot_compare(trueValue, predictValue)

	# np.savetxt('test_result4.txt',test_result)
	# np.savetxt('train_result4.txt',train_result)
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
