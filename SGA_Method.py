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
from multiprocessing import Process
from functools import reduce
import pandas as pd
import tensorflow as tf
from copy import copy

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

参数编码,使用实数编码
初始群体的设计
适应度函数的设计
遗传操作设计(选择,交叉,变异)
控制参数的设定

'''
"""
获取到的结果放到队列中，共其他主机使用
"""
"""
使用遗传算法来优化参数，然后再进行训练，最后再实现tensorflows的多机并行。
利用染色体的每一位来表示阀值或权值。然后确定种群大小，即一个种群有多少个个体。确定这些之后就可以进化物种
所谓使用的遗传算法来优化神经网络，其实是优化神经网络的初始节点值，不再是随机的初始值。
利用染色体中的每一位来代替LSTM神经网络的权值和阀值
"""
"""
思路：
    设置种群数目和优化目标,对神经元初始值与阀值进行实数编码，然后计算各种群适应度，
    选择操作，交叉操作，变异操作，如果没有达到优化目标，则继续上面的步骤；否则将阀值和权值传入到神经网络，
    计算误差，权值和阀值更新，是否满足结束条件,如果没有，则继续上面的步骤，否则进行模型的准确性的验证。
"""
#读取数据
def load_data(filename):
    return np.loadtxt(filename)


class SGA(object):
    """
    一个简单的遗传算法来实现LSTM在不同机器之间的运行，加速运行时间
    """
    def __init__(self,population_size,max_gen,crossover,mute,input_num,hidden_num):
        """
        :param population_size: 种群大小
        :param max_gen: 最大迭代数
        :param crossover: 交叉概率
        :param mute: 变异概率
        :param binary_num: 变量的二进制位数
        :param gap: 代沟
        :param hidden_num:隐层数目
        :param input_num:输入层数目
        """
        self.population_size=population_size
        self.max_gen=max_gen
        self.crossover=crossover
        self.mution=mute
        self.best=[0]*max_gen
        self.input_num=input_num
        self.hidden_num=hidden_num
        pass

    def initial_pop(self):
        """
        初始化种群
        :return:
        """
        pass

    def select(self,pop_size,individual):
        """
        :param pop_size: 种群规模
        :param individual: 种群信息
        :return: 返回选择后的新种群
        """
        #step_1:求适应度值的倒数
        fitness_val=[]  #存储当前种群所有的个体适应值
        fitness_val=self.fitness_fun(indivaduals=None)

        reverse_fitness_val=[]  #用于存储每个个体适应度的倒数,10表示系数
        for i in range(len(fitness_val)):
            reverse_fitness_val.append(10./fitness_val[i])

        #step_2:个体选择概率
        sum_reverse_fitness_val=sum(reverse_fitness_val)
        #存储每一个个体被选择的概率
        per_individual_select_pro=[]
        for i in range(len(sum_reverse_fitness_val)):
            per_individual_select_pro.append(reverse_fitness_val[i]/sum_reverse_fitness_val)

        #step_3:采用轮盘赌法选择新的个体
        index=[] #用于存储适应度高个体的索引下标
        for i in range(self.population_size):
            pick=np.random.rand()  #随机产生一个[0,1)的随机数
            while pick==0:
                pick=np.random.rand()
            for j in range(self.population_size):
                pick=pick-per_individual_select_pro[j]
                if pick<0:
                    index.append(j) ###
                    break

        #step_4:得到被选择的优秀个体的适应度值和及对应的实数编码值
        # new_chrom=[[],[]]  #第一个存储染色体编码值,第二个存储对应的的适应度值
        new_chrom={
            "chorm":[],
            "fitness_val":[]
        }
        for i in range(len(index)):
            new_chrom["chorm"].append(index[i])
            new_chrom["fitness_val"].append(fitness_val[index[i]])

        return new_chrom

    def cross(self,lenchrom,chrom,bound,pop_size,p_cross):
        """
        :param lenchrom: 染色体长度
        :param chrom: 染色体种群
        :param bound: 边界大小
        :param pop_size: 种群规模
        :param p_cross: 交叉概率
        :return:
        """
        index=[] #存储要交叉的染色体下标
        for i in range(pop_size):
            #随机生成两个数
            pick=np.random.rand(1,2)
            while (pick[0][1]*pick[0][0])==0:
                pick=np.random.rand(1,2)

            #存储要交叉的染色体的下标
            for j in range(2):
                index.append(pick[0][j]*pop_size)

            #交叉概率决定是否交叉
            pick=np.random.rand()
            while pick==0:
                pick=np.random.rand()

            if pick>p_cross:
                continue

            #利用一个标志位
            flag=0
            #随机选择交叉位
            while flag==0:
                pick=np.random.rand()
                while pick==0:
                    pick=np.random.rand()
            #随机选择进行交叉的位置,即选择第几个变量进行交叉,两个染色体交叉的位置相同
                cross_pos=np.ceil(pick*sum(lenchrom))   #这个还没写好
                pick=np.random.rand()
                #采用的是实数交叉法
                v1=chrom[index[0],cross_pos]
                v2=chrom[index[1],cross_pos]
                chrom[index[0],cross_pos]=pick*v2+(1-pick)*v1
                chrom[index[1],cross_pos]=pick*v1+(1-pick)*v2
                #交叉结束
                # 需要判断染色体的可行性,如果可行,就返回交叉后的染色体,如果不可以则继续进行交叉
                # 写一个测试染色体的可行的代码test()
                flag_1=test()
                flag_2=test()
                if flag_1==0 and flag_2==0:
                    flag=0
                else:
                    flag=1
        return chorm


    #变异操作
    def mutation(self,num,chrom,bound,lenchrom,pop_size,p_mute):
        """
        :param num: 当前的迭代次数
        :param chrom: 染色体种群
        :param bound: 边界
        :param lenchrom: 染色体长度
        :param pop_size: 种群规模
        :param p_mute: 变异率
        :return: 返回变异后的染色体
        """
        for i in range(pop_size):
            #随机选择一个染色体进行变异
            pick=np.random.rand()
            while pick==0:
                pick=np.random.rand()
            index=np.ceil(pick*pop_size)

            #变异概率决定该轮循环是否进行变异
            pick=np.random.rand()
            if pick>p_mute:
                continue

            #利用一个标志来判断这次变异是否可行
            flag=0
            while flag==0:
                pick=np.random.rand()
                while pick==0:
                    pick=np.random.rand()
                #随机选择了染色体变异的位置,即选择了第pos个变量进行变异
                mute_pos=np.ceil(pick*sum(lenchrom))
                #变异开始
                pick_mute=np.random.rand()
                fg=np.random.rand()*(1-num/self.max_gen)**2
                if pick>0.5:
                    #chorm表示种群中染色体的集合
                    chrom[i,pos]=chrom[i,pos]+(chrom[i,pos]-bound[pos,2])*fg
                else:
                    chrom[i,pos]=chrom[i,pos]+(bound[pos,1]-chrom[i,pos])*fg

                #验证解的可行性
                flag=test()
        return chrom

    def reshape_weight_bias(self,indivduals):
        """
        :param indivduals:
        :return:
        """
        w1 = indivduals[0:self.compute_param_num()[1]]
        b1 = indivduals[self.compute_param_num()[1]:self.compute_param_num()[0]]

        new_w1=np.reshape(w1,newshape=(1,self.hidden_num))
        new_b1=np.reshape(b1,newshape=(self.hidden_num))

        return new_w1,new_b1

    def fitness_fun(self,indivaduals):
        """
        适应度函数选择使用预测误差的范数,返回该个体的适应度值
        计算染色体的适应度值,然后将其进行排序
        直接在这里将网络导入进来计算适应度值
        :return:
        """
        #step_1:提取参数,w1,b1
        for i in range(self.population_size):
            new_w1,new_b1=self.reshape_weight_bias(indivaduals[i])
        #step_3:网络权值赋值

        #step_4:网络训练,计算误差,误差表示个体的适应度值
        #赋值之后进行网络训练
        #计算误差
        #return error=np.sum(np.abs(y-pre_y) 所有染色体的适应度值.用一个列表表示
        #return []

    def bestfitness(self,individual):
        """
        计算每一代种群的最佳适应度,以及对应的索引下标
        :return:
        """
        fitness_val=[]
        fitness_val=self.fitness_fun(indivaduals=None)
        return min(fitness_val),np.argmin(fitness_val,axis=0)

    def avgfitness(self,individual):
        """
        计算每一代种群的平均适应度
        :return:
        """
        fitness_val=[]
        fitness_val=self.fitness_fun(indivaduals=None)
        return np.average(fitness_val,axis=0)

    def compute_param_num(self):
        """
        返回参数之和
        :return:
        """
        w1_num=self.input_num*self.hidden_num
        b1_num=self.hidden_num
        return w1_num+b1_num,w1_num,b1_num

    def code(self,len_chrom,bound):
        """
        本函数将变量编码变成染色体,用于随机初始化一个种群
        :param lenchrom: 染色体长度
        :param bound: 变量的取值范围
        :return: 返回染色体的编码值,染色体长度,边界范围
        """
        pick=np.random.rand(size=(1,len_chrom.shape[1]))
        #线性插值,编码结果以实数向量存入ret中。
        ret=bound[:,1]+(bound[:,2]-bound[:,1])*pick
        return ret,len_chorm,bound

    def test(self):
        """
        测试染色体是否可行,可行就返回1,反之则返回0
        :return:
        """
        pass

#程序的入口文件
if __name__=="__main__":
    dataset = load_data("power_data.txt")
    # 提取相应的测试数据和训练数据
    # 对数据进行预处理
    # 构建网络
    # 遗传算法参数初始化
    max_gen = 100
    pop_size = 30
    pcross = 0.3
    pmutation = 0.1

    # 节点总数
    input_num=1
    hidden_num=4
    sga=SGA(population_size=pop_size,max_gen=max_gen,crossover=pcross,mute=pmutation,input_num=input_num,hidden_num=hidden_num)
    sum_num,w_num,b_num=sga.compute_param_num()
    len_chrom=np.ones(shape=(1,sum_num))
    bound=[-3*np.ones(shape=(len_chrom.shape[1],1)),3*np.ones(shape=(len_chrom.shape[1],1))]

    #定义种群信息,适应度值和染色体种群
    individuals={
        "fitness":np.zeros(1,pop_size),
        "chrom":[]
    }
    #每一代种群的最佳适应度值
    bestfitness=[]
    #每一代种群的平均适应度值
    avgfitness=[]
    #适应度最好的染色体
    bestchrom=[]

    #初始化种群
    for i in range(pop_size):
        #随机产生一个种群
        individuals["chrom"].append(sga.code(len_chrom=len_chrom,bound=bound)) #编码
        x=individuals["chrom"][i]
        #计算每个个体的适应度值
        individuals["fitness"][i]=sga.fitness_fun(indivaduals=individuals)  #适应度值

    #找最好的染色体,以及对应的下标
    best_index,bestfitness=np.argmin(individuals["fitness"],axis=0),np.min(individuals["fitness"],axis=0)
    #最好的染色体
    bestchrom=individuals["chrom"][best_index]

    #迭代求解最佳的初始阀值和权值
    #进化开始
    for i in range(max_gen):
        #选择
        individuals=sga.select(pop_size=pop_size,individual=individuals)
        #交叉
        individuals["chrom"]=sga.cross(p_cross=pcross,lenchrom=len_chrom,chrom=individuals["chrom"],bound=bound,pop_size=pop_size)
        #变异
        individuals["chrom"]=sga.mutation(num=i,chrom=individuals["chrom"],bound=bound,lenchrom=len_chrom,pop_size=pop_size,p_mute=pmutation)

        #继续计算适应度
        for j in range(pop_size):
            x=individuals["chrom"][j]
            individuals["fitness"][j]=sga.fitness_fun(indivaduals=individuals["chrom"])
    #找到最小和最大适应度的染色体以及它们在种群中的位置
        newbest_fitness,newbest_index=np.min(individuals["fitness"],axis=0),np.argmin(individuals["fitness"],axis=0)
        worset_fitness,worest_index=np.max(individuals["fitness"],axis=0),np.argmax(individuals["fitness"],axis=0)
        #代替上一次进化中最好的染色体
        if bestfitness>newbest_fitness:
            bestfitness=newbest_fitness
            best_index=newbest_index
            bestchrom=individuals["chrom"][best_index]
        individuals["chrom"][worest_index]=bestchrom
        individuals["fitness"][worest_index]=bestfitness

    x=bestchrom

    #把最优初始阀值权值赋予给网络预测
    #获取最优网络权值和阀值，并且改变形状以适应神经网络训练的需求
    new_w1,new_b1=sga.reshape_weight_bias(x)

    #将最优权值保存起来,以便在LSTM神经网络下进行训练
    with open("w_b_data.txt","w") as f:
        f.write(new_w1+"\n")
        f.write(new_b1+"\n")



















