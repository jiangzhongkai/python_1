# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

f = open('./power_data.txt','rb')
data = []
for line in f.readlines():
	line = int(line.strip())
	data.append(line)
# nums= len(data)/750.0 	# 46.72


data = data[480:]		# 480, 因为1997年的前五天是从周三开始算的，故选择舍弃掉前5天的数据，按照1天96个数据，96*5=480
dataset = []
# for i in range(0, len(data)-750, 750):	# 减去750，是因为数据最后一个区间无法达到750，所以干脆舍弃掉最后一个区间
# 	sequence = []
# 	sequence = data[i:i+750]
# 	dataset.append(sequence)


for i in range(0, len(data)-672, 672):	# one hour is 4 point, then a day is need 24*4=96 point, so a week is 96*7=672 point
	sequence = data[i:i+672]
	dataset.append(sequence)


dataset = np.array(dataset)
print(len(dataset))


# for i in range(51):
# 	plt.plot(dataset[i])
# 	plt.show()

# plt.plot(dataset[5])
# plt.show()

# for i in range(11, 13):
# 	plt.plot(dataset[i])
# 	plt.show()

f = open('./power_label.txt','r')
label = []
for line in f.readlines():
	line = line.strip()
	line = int(line)
	label.append(line)

print(label)
# label = np.array(label)
# normal_data = dataset[label==1]
# print len(normal_data)

# time_steps = normal_data.shape[1]
# data_x = normal_data[:,:time_steps-1]
# data_y = normal_data[:,1:]
# print data_y.shape
# sh = data_x.shape
# sh = sh + (1,)
# print sh
# data_x = data_x.reshape(sh)
# data_y = data_y.reshape(sh)
# print data_y.shape
# testSet = dataset[label==0]
# print testSet.shape
# test_x = np.mat(testSet)[0]
# print test_x.shape




# 方法2
# 周期由一个星期改为一天


# 下采样
# 将时间序列中每８个值用１个值代替，这里选择８个值的均值
# 之前是15分钟一个值，一天96个值，另672代表一周的数据 96*7=672
# 现在降采样后，2小时一个值，一天12个值
downSample_data = []
for i in range(len(dataset)):
	line = np.ones([int(672/8)])
	j = 0
	for start, end in zip(range(0,672,8), range(8,673,8)):
		temp_array = dataset[i,start:end]
		temp_array = np.mean(temp_array)
		line[j] = temp_array
		j = j + 1
	downSample_data.append(line)

downSample_data = np.array(downSample_data)
print(downSample_data.shape)

# plt.plot(downSample_data[11])
# plt.show()
# print downSample_data[50]

# for i in range(51):
# 	plt.plot(downSample_data[i])
# 	plt.show()

def MinMaxScaling(data):	
	scaler = MinMaxScaler(feature_range=(0, 1))
	data_scaling = scaler.fit_transform(data)
	return data_scaling, scaler


# dataSet, scaler = MinMaxScaling(downSample_data)
# print dataSet.shape

# for i in range(20):
# 	plt.plot(dataSet[i])
# 	plt.show()

# dataSet = scaler.inverse_transform(dataSet)
# for i in range(20):
# 	plt.plot(dataSet[i])
# 	plt.show()

# plt.hist(downSample_data[i])
# plt.show()


'''
扩展数据集
通过将一条长度为672的电量时间序列，分离成8条长度为84的时间序列
这与之前的下采样是不同的。主要思想如下：
1.首先计算分离成8条时间序列，每条时间序列的长度
2.针对每一条672长度的时间序列，将每8个值，进行分离，第一条序列得第1个值，第二条序列得第2个值，
	以此类推，第八条序列得第8个值。
由此就扩充了数据集，而不是单独得取平均值，舍弃掉了一些信息。
函数内容大概如下：
循环遍历原始数据集中每一个时间序列，其长度为672
	计算分离成8条时间序列，每条时间序列的长度
	创建一个形状为8X84的数据，等下用来装8条84长度的时间序列
	循环遍历8次，代表8条时间序列：
		原数据集角标直接用i来初始化，意思就是第一次从0开始进行接下来的for循环，第二次从1开始
		循环84次，每一轮对某i序列进行赋值，赋值完后，同时把原始数据集角标往后移动8位
			意思就是，第一轮取0，第二轮取8，第三轮取16，以此类推，共84轮
'''
def extendeDataset(dataSet):
	extendeData = []
	for m in range(len(dataSet)):
		size = int(672/8)	# 计算分离成8条时间序列，每条时间序列的长度
		# 创建一个形状为8X84的数据，即存储着8条84长度的时间序列		
		data = np.zeros([8,size])
		for i in range(8):
			index = i
			for j in range(size):
				data[i,j] = dataSet[m,index]
				index = index + 8	# 把原始数据集角标往后移动8位
		extendeData.extend(data)	# 使用extend函数而不是append函数，使得其直接成为二维数组
	print(np.array(extendeData).shape) #（408,84）,408=51*8
	return np.array(extendeData)


newdata = extendeDataset(dataset)
newLabel = np.array(label).repeat(8)
print(newLabel.shape)
normalData = newdata[newLabel==1]
abnormalData = newdata[newLabel==0]

print(normalData.shape)		# (352, 84)
print(abnormalData.shape)	# (56, 84)

# for i in range(len(abnormalData)):
# 	plt.plot(abnormalData[i])
# 	plt.show()
