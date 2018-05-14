import numpy as np
import matplotlib.pyplot as plt
from power_demand2 import rnn_data

'''

本代码文件是针对power_demand2预测的结果，将分别对训练集和测试集的真实数据和预测数据进行绘制的

'''



train_predict = np.loadtxt('./train_result7.txt')
test_preditc = np.loadtxt('./test_result7.txt')


# 由于之前输入模型时，前面timestep个数据并没有输入到模型中，被舍去了，
# 故而现在在恢复预测结果时，也要删去剩下的（84-timestep）个数据，保持真实数据与预测数据同步
# 例如下面舍去前面84-20=64个数据，672/8=84代表下采样后一周的数据
def data_recovery(dataset, timesteps=20):
	dataset = dataset[84 - timesteps:]
	dataset = dataset.reshape([-1,84])
	return dataset


train_predict = dataFilling(train_predict)

test_preditc = dataFilling(test_preditc)

train_x, train_y, test_x, test_y, scaler = rnn_data('./power_data.txt')
train_y = scaler.inverse_transform(train_y)
test_y = scaler.inverse_transform(test_y)

train_y = dataFilling(train_y)
test_y = dataFilling(test_y)

print test_y[-1]


def plotpredit(dataset):
	for i in range(len(dataset)):
		plt.plot(dataset[i])
		plt.show()

# plotpredit(train_predict)
# plotpredit(test_preditc)



def plotCompare(dataset1, dataset2):
	for i in range(len(dataset1)):
		plt.plot(dataset1[i], label='predict')
		plt.plot(dataset2[i], label='true')
		plt.legend(loc = 'upper right')
		plt.show()

# plotCompare(test_preditc, test_y)
plotCompare(train_predict, train_y)
