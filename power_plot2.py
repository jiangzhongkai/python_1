# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt 
import power_demand4 as pod4
import power_demand5 as pod5
import power_demand6 as pod6



# 图像比较
def plot_compare(dataset1, dataset2):
	for i in range(len(dataset1)):
		plt.plot(dataset1[i], label='true')
		plt.plot(dataset2[i], label='predict')
		plt.legend(loc = 'upper right')
		plt.show()


def plot_compare2(dataset1, dataset2):
	for i in range(len(dataset1)):
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		ax1.plot(dataset1[i], label='true')
		ax2.plot(dataset2[i], label='predict', color='green')
		ax1.legend(loc = 'upper right')
		ax2.legend(loc = 'upper right')
		plt.show()



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


def calculateMSE(trueValue, predictValue, norm):
	# 把预测的结果进行数据还原
	predictValue = inverseNorm(predictValue, norm)
	print(predictValue.shape)
	# 真实的结果也需要还原，因为之前真实的数据也进行了归一化
	trueValue = inverseNorm(trueValue, norm)
	# MSE = np.mean(np.sum(np.square(trueValue - predictValue))) BIG ERROR
	MSE = np.mean(np.square(trueValue - predictValue))
	print(MSE)
	return MSE, trueValue, predictValue



if __name__ == '__main__':
	_, train_y, _, test_y, train_norm, test_norm = pod6.rnn_data('./power_data.txt')
	# 拿到之前模型的预测结果，train_result表示为训练集输入模型得到的预测结果
	train_result = np.loadtxt('./train_result.txt')
	test_result = np.loadtxt('./test_result.txt')

	train_true = inverseNorm(train_y, train_norm)
	train_predict = inverseNorm(train_result, train_norm)

	test_true = inverseNorm(test_y, test_norm)
	test_predict = inverseNorm(test_result, test_norm)

	# plot_compare(train_true, train_predict)
	# print test_predict.shape
	plot_compare2(test_true, test_predict)
