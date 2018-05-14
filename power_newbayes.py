# coding: utf-8
"""
这一部分是属于bayes部分
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.metrics import roc_curve
from power_bayes3_1 import createErrDataset,splitErrDataset
import power_demand6 as pod6
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cross_validation import train_test_split

# from sklearn.preprocessing import train_test_split


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


if __name__ == "__main__":
    _, train_y, _, test_y, train_norm, test_norm = pod6.rnn_data('./power_data.txt')

    # 拿到之前模型的预测结果，train_result表示为训练集输入模型得到的预测结果
    train_result = np.loadtxt('./train_result.txt')
    test_result = np.loadtxt('./test_result.txt')
    # 恢复
    train_true = inverseNorm(train_y, train_norm)
    train_predict = inverseNorm(train_result, train_norm)

    test_true = inverseNorm(test_y, test_norm)
    test_predict = inverseNorm(test_result, test_norm)

    normal_error = np.abs(train_true-train_predict)
    abno_error = np.abs(test_true-test_predict)

    print("normal_error.shape",normal_error.shape)
    print("abno_error.shape",abno_error.shape)

    normal_error = np.c_[normal_error, np.zeros(len(normal_error))]
    abno_error = np.c_[abno_error, np.ones(len(abno_error))]


    dataset = np.r_[normal_error, abno_error]
    np.random.shuffle(dataset)

    train_x, test_x, train_y, test_y = train_test_split(dataset[:,:-1], dataset[:,-1], test_size=0.3, random_state=42)


    clf = GaussianNB()
    clf.fit(train_x, train_y)
    y_hat = clf.predict(train_x)
    y_score = clf.predict_proba(train_x)
    y_log_score = clf.predict_log_proba(train_x)
    y_test_hat = clf.predict(test_x)
    y_test_score = clf.predict_proba(test_x)
    print(accuracy_score(train_y, y_hat))
    print(metrics.recall_score(train_y, y_hat))
    print(metrics.classification_report(train_y, y_hat))
    print(metrics.classification_report(test_y, y_test_hat))
    print(y_score)
    print(y_test_score)
    print(y_test_hat)
    print(clf.classes_)

    # fpr, tpr, thresholds = metrics.roc_curve(train_y, y_hat)
    fpr, tpr, thresholds = metrics.roc_curve(test_y, y_test_score[:,-1])
    print(fpr, tpr, thresholds)

    # fpr = [0, 0.1,  0.4,  0.5,  0.8,  1]
    # tpr = [0, 0.88, 0.90, 0.94, 0.98, 1]
    plt.plot(fpr, tpr, c='#FF0902', lw=2, alpha=0.7)
    plt.plot((0, 1), (0, 1), c='#808080', lw=2, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.show()
