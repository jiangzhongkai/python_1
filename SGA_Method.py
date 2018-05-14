"""-*- coding: utf-8 -*-
 DateTime   : 2018/5/14 12:34
 Author  : Peter_Bonnie
 FileName    : SGA_Method.py
 Software: PyCharm
"""
import os
import re

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

    def initial_pop(self,):
        pass

    def cross(self):
        pass
    def mute(self):
        pass
