#　coding: utf-8
import numpy as np


def createDataSet(filename):
	f = open(filename,'rb')
	data = []
	for line in f.readlines():
		line = int(line.strip())
		data.append(line)
	# nums= len(data)/750.0 	# 46.72

 
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
		line = np.ones([int(672/8)])
		j = 0
		for start, end in zip(range(0,672,8), range(8,673,8)):
			temp_array = dataSet[i,start:end]
			temp_array = np.mean(temp_array)
			line[j] = temp_array
			j = j + 1
		downSample_data.append(line)
	downSample_data = np.array(downSample_data)
	return downSample_data


# 随机创建质心函数
def randCent(dataSet, k):
	# dataMat = np.mat(dataSet)
	n = dataSet.shape[1]
	# 创建k个点作为起始质心
	centroid = np.zeros([k,n])		# change centroid to a matrix
	for j in range(n):
		minj = np.min(dataSet[:,j])
		rangej = np.max(dataSet[:,j]) - minj
		# 之前发生错误
		# ValueError: could not broadcast input array from shape (3,1) into shape (3)
		# because centroid[:,j].shape=(k,)  so need change centroid to a matrix
		centroid[:,j] = minj + rangej * np.random.rand(k)
	return centroid



def distance(a, b):
	return (a-b)*(a-b)



def distDtw(vec1, vec2):
	# create distance matrix
	# print type(vec2)
	m, n = len(vec1), len(vec2)
	dist = np.zeros([m+1,n+1])
	for i in range(1,m+1):
		for j in range(1,n+1):
			dist[i][j] = distance(vec1[i-1], vec2[j-1])

	D = np.zeros([m+1, n+1])
	for i in range(1, m+1):
		for j in range(1, n+1):
			D[i][j] = dist[i][j] + min(D[i-1][j], D[i][j-1], D[i-1][j-1])
	return D[m][n]



# 计算质心－分配－重新计算，反复迭代
def kMeans(dataSet, k, distCalculate=distDtw, createCent=randCent):
	m = dataSet.shape[0]
	# labels数组是mx2形状的数组，其每一行存放着每一个样本属于相应的类别的编号
	# 及距离那个类别的距离，一共m个样本，所以m行
	labels = np.zeros([m,2])
	# create the centroid
	centroid = createCent(dataSet, k)	# 之前发生错误是因为这里写成了dataset了，导致古怪的错误
	# 聚类是否改变的标志，先设置flag=True
	flag = True
	# 当labels里的值不再改变时，就终止循环
	while flag:
		flag = False
		# 1. 每一个样本分配到离他最近的质心堆
		for i in range(m):
			minDist = np.inf
			minIndex = -1
			for j in range(k):
				dist = distCalculate(dataSet[i], centroid[j])
				if dist < minDist:
					minDist = dist; minIndex = j
			# 当labels里的值不再改变时，就终止循环
			if labels[i,0]!= minIndex:
				flag = True
			labels[i,:] = minIndex, minDist**2
		# print centroid
			
		#　2. 等到所有的样本经过一轮的分配后，重新计算质心，根据每一堆的平均值来得到质心
		for cent in range(k):
			# 获取相同一个类别的角标
			indice = np.nonzero(labels[:,0] == cent)[0]
			# print 'indice=',indice
			
			# 得到相同类别的数据
			tmp = dataSet[indice]
			# print 'tmp=',tmp.shape
			
			# 求平均数
			centroid[cent,:] = np.mean(tmp, axis=0)

	return centroid, labels

if __name__ == '__main__':
	dataSet = createDataSet('./power_data.txt')
	dataSet = downSample(dataSet)
	centroid, labels = kMeans(dataSet, 2)
	np.savetxt('labels2.txt',labels)
