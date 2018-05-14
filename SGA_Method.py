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
1.种群大小->染色体->基因变化
"""

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
      self.bound=bounds
      pass

    def initial_pop(self):
        """
        :return:
        """

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


if __name__=="__main__":
    lstm_sga=SGA(cross_p=0.2,mute_p=0.01,population_size=100000,pop_ac=1e-8)
    lstm_sga.initial_pop()
    pass
