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

class SGA(object):
    """
    一个简单的遗传算法来实现LSTM在不同机器之间的运行，加速运行时间
    """
    def __init__(self,cross_p,mute_p,population_size,pop_ac):
        """
        :param cross_p: 交叉概率
        :param mute_p: 变异概率
        :param population_size:种群大小
        :param pop_ac: 精确度
        """
        self.cross_p=cross_p
        self.mute_p=mute_p
        self.population_size=population_size
        self.pop_ac=pop_ac
        pass

    def initial_pop(self):
        """
        初始化种群
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