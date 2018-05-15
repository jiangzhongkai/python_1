"""-*- coding: utf-8 -*-
 DateTime   : 2018/5/14 12:34
 Author  : Peter_Bonnie
 FileName    : SGA_Method.py
 Software: PyCharm
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

"""
思想:
1.种群大小->染色体->基因
2.写一个生成染色体类
3.写一个遗传算法类,
4.将染色体类生成的多个染色体放到种群中去
5.利用遗传算法寻找最优解能否减少寻解的时间？
6.通过遗传算法获得结果传给神经网络是不是会更加快呢？

choose an intial population
determine the fitness of each individual
perform selection
repeat
    perform crossover
    perform mutation
    determine the fitness of each individual
    perform selection
until some stopping criterion applies

====
将遗传算法运用神经网络的难点就是染色体该如何编码？以及评价函数该如何选择？
   =====辅助结合方式：用GA对数据进行预处理，然后用NN解决问题，例如模式识别用GA进行特征提取，再用nn进行分类
   =====合作，GA和NN共同处理问题，在NN固定的网络拓扑下，利用GA确定链接权重，或者直接利用GA优选网络结构再用，bp训练网络
"""
'''
交叉,变异，适应度，种群，染色体，基因，选择

参数编码
初始群体的设计
适应度函数的设计
遗传操作设计(选择,交叉,变异)
控制参数的设定

'''


class SGA(object):
    """
    一个简单的遗传算法来实现LSTM在不同机器之间的运行，加速运行时间
    """
    def __init__(self,cross_p,mute_p,population_size,precsion,max_generate,bounds):
      """
      :param cross_p: 交叉概率
      :param mute_p: 变异概率
      :param population_size: 种群大小
      :param precsion: 精确度
      :param max_generate: 最大的迭代次数
      :param bound:变量范围
      """""
      self.cross_p=cross_p
      self.mute_p=mute_p
      self.population_size=population_size
      self.precsion=precsion
      self.max_generate=max_generate
      self.bounds=bounds

      self.best=[0]*max_generate
      self.pop=[]

      pass

    def initial_pop(self):
        """
        初始化种群
        :return:
        """
        for i in range(self.max_generate):
            #生成一个染色体，然后将染色体放到列表中去
            rand_x=random.rand(0,1)
            self.pop.append("染色体")


        pass

    def update_bestSolution(self,solution,bestsolution):
        """
        更新最佳的解决方案
        :param solution:原始的方案
        :param bestsolution: 最佳的方案
        :return:
        """
        pass

    def select_pop(self):
        """
        选择相应的种群来进行交叉
        :return:
        """
        pass

    def roll_select(self):
        """
        使用轮盘赌的原则来选择交叉的染色体
        :return:
        """

        pass

    def cross(self):
        """
        进行交叉
        :return:
        """
        pass

    def mute(self):
        """
        进行变异
        :return:
        """


        pass

class Choromones(object):
    """
    生成染色体
    """
    def __init__(self):
        """

        """
        pass

if __name__=="__main__":
    lstm_sga=SGA(cross_p=0.2,mute_p=0.01,population_size=100000,pop_ac=1e-8)
    lstm_sga.initial_pop()
    pass
