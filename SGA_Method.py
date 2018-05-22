"""-*- coding: utf-8 -*-"""
 #DateTime   : 2018/5/14 12:34
 #Author  : Peter_Bonnie
 #FileName    : SGA_Method.py
 #Software: PyCharm

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
from LSTM import *


"""
思想:
1.种群大小->染色体->基因
2.写一个生成染色体类
3.写一个遗传算法类,
4.将染色体类生成的多个染色体放到种群中去
5.利用遗传算法寻找最优解能否减少寻解的时间？
6.通过遗传算法获得结果传给神经网络是不是会更加快呢？
交叉,变异，适应度，种群，染色体，基因，选择
参数编码,使用实数编码
初始群体的设计
适应度函数的设计
遗传操作设计(选择,交叉,变异)
控制参数的设定
"""
"""
思路：
    使用遗传算法来优化参数，然后再进行训练，最后再实现tensorflows的多机并行。
    利用染色体的每一位来表示阀值或权值。然后确定种群大小，即一个种群有多少个个体。确定这些之后就可以进化物种
    所谓使用的遗传算法来优化神经网络，其实是优化神经网络的初始节点值，不再是随机的初始值。
    利用染色体中的每一位来代替LSTM神经网络的权值和阀值 
    设置种群数目和优化目标,对神经元初始值与阀值进行实数编码，然后计算各种群适应度，
    选择操作，交叉操作，变异操作，如果没有达到优化目标，则继续上面的步骤；否则将阀值和权值传入到神经网络，
    计算误差，权值和阀值更新，是否满足结束条件,如果没有，则继续上面的步骤，否则进行模型的准确性的验证。
"""

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


    def select(self,pop_size,individuals):
        """
        :param pop_size: 种群规模
        :param individuals: 种群信息
        :return: 返回选择后的新种群
        """
    #======================================================
        #采用的轮盘赌选择原则
    #####################################################
        #step_1:初始化种群
        fitness_val=[]
        # step_2:计算每个个体的适应度
        for i in range(pop_size):
            temp_fitness_val=self.fitness_fun(individuals[i])
            #由于这里适应误差值作为适应度值,所以越小越好
            temp_fitness_val=1.0/temp_fitness_val
            fitness_val.append(temp_fitness_val)
        #step_3:计算所有染色体的适应度以及染色体被选择的概率
        per_individual_select_prob=[]
        for i in range(len(fitness_val)):
            per_individual_select_prob.append(fitness_val[i]/sum(fitness_val))

        #step_4:计算种群中每个个体的被选中的累积概率
        caculate_prob=[]
        temp_sum_prob=[]  #临时存储每个个体的累计概率
        for j in range(len(fitness_val)):
            temp_sum_prob.append(per_individual_select_prob[j])
            caculate_prob.append(sum(temp_sum_prob))

        #step_5:根据累计概率来选择染色体
        new_chrom={
            "fitness":[],
            "chrom":[]
        }
        for i in range(pop_size):
            pick=np.random.rand()
            while pick==0:
                pick=np.random.rand()

            for j in range(len(caculate_prob)):
                if pick<caculate_prob[0]:
                    new_chrom["chrom"].append(individuals[0])
                    new_chrom["fitness"].append(fitness_val[0])
                    break
                elif pick<=caculate_prob[j] and pick>=caculate_prob[j-1]:
                    new_chrom["chrom"].append(individuals[j])
                    new_chrom["fitness"].append(fitness_val[j])
                    break
        return new_chrom

        # #step_1:求适应度值的倒数
        # fitness_val=[]  #存储当前种群所有的个体适应值
        # #计算每个个体的适应度值
        # for i in range(pop_size):
        #     fitness_val.append(self.fitness_fun(individuals=individual["chrom"][i]))
        #
        # reverse_fitness_val=[]  #用于存储每个个体适应度的倒数,10表示系数
        # for i in range(len(fitness_val)):
        #     reverse_fitness_val.append(10./fitness_val[i])
        #
        # #step_2:个体选择概率
        # sum_reverse_fitness_val=sum(reverse_fitness_val)
        # #存储每一个个体被选择的概率
        # per_individual_select_pro=[]
        # for i in range(len(sum_reverse_fitness_val)):
        #     per_individual_select_pro.append(reverse_fitness_val[i]/sum_reverse_fitness_val)
        #
        # #step_3:采用轮盘赌法选择新的个体
        # index=[] #用于存储适应度高个体的索引下标
        # for i in range(pop_size):
        #     pick=np.random.rand()  #随机产生一个[0,1)的随机数
        #     while pick==0:
        #         pick=np.random.rand()
        #     for j in range(pop_size):
        #         pick=pick-per_individual_select_pro[j]
        #         if pick<0:
        #             index.append(j) ###
        #             break
        # #step_4:得到被选择的优秀个体的适应度值和及对应的实数编码值
        # new_chrom={
        #     "chrom":[],
        #     "fitness":[]
        # }
        # #返回新的染色体
        # for i in range(len(index)):
        #     new_chrom["chorm"].append(individual["chrom"][index[i]])
        #     new_chrom["fitness"].append(individual["fitness"][0][index[i]])
        # return new_chrom

    def cross(self,len_chrom,chrom,bound,pop_size,p_cross):
        """
        :param lenchrom: 染色体长度
        :param chrom: 染色体种群
        :param bound: 边界大小
        :param pop_size: 种群规模
        :param p_cross: 交叉概率
        :return:
        """
        index=[] #存储要交叉的染色体下标
        for i in range(pop_size-1):
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
                cross_pos=np.ceil(pick*sum(len_chrom))   #这个还没写好
                pick=np.random.rand()
                #采用的是实数交叉法
                v1=chrom[index[0]][cross_pos]
                v2=chrom[index[1]][cross_pos]

                chrom[index[0]][cross_pos]=pick*v2+(1-pick)*v1
                chrom[index[1]][cross_pos]=pick*v1+(1-pick)*v2
                # v1=chrom[index[0],cross_pos]
                # v2=chrom[index[1],cross_pos]
                # chrom[index[0],cross_pos]=pick*v2+(1-pick)*v1
                # chrom[index[1],cross_pos]=pick*v1+(1-pick)*v2
                #交叉结束
                # 需要判断染色体的可行性,如果可行,就返回交叉后的染色体,如果不可以则继续进行交叉
                # 写一个测试染色体的可行的代码test()
                #只需要对新产生的染色体进行检查是否可行
                flag_1=self.test(bound=bound,chrom=chrom[index[0]][cross_pos])
                flag_2=self.test(bound=bound,chrom=chrom[index[1]][cross_pos])
                if flag_1*flag_2==0:
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
                    chrom[index][mute_pos]=chrom[index][mute_pos]+(chrom[index][mute_pos]-bound[1][0])*fg
                    # chrom[i,pos]=chrom[i,pos]+(chrom[i,pos]-bound[pos,2])*fg
                else:
                    chrom[index][mute_pos]=chrom[index][mute_pos]+(bound[0][0]-chrom[index][mute_pos])*fg
                    # chrom[i,pos]=chrom[i,pos]+(bound[pos,1]-chrom[i,pos])*fg

                #验证解的可行性
                flag=self.test(bound=bound,chrom=chrom[index][mute_pos])
        return chrom  #返回一个种群


    def reshape_weight_bias(self,individual):
        """
        :param indivdual[][]:
        :return:
        """
        w1 = individual[0][0:self.compute_param_num()[1]]
        b1 = individual[0][self.compute_param_num()[1]:self.compute_param_num()[0]]
        new_w1=np.reshape(w1,newshape=[1,4]).astype(dtype=np.float32)
        new_b1=np.reshape(b1,newshape=[4]).astype(dtype=np.float32)
        print(new_w1)
        # print(type(new_w1))
        return new_w1,new_b1


    #对每一个个体的适应度进行求值
    def fitness_fun(self,individual):
        """
        适应度函数选择使用预测误差的范数,返回该个体的适应度值
        计算染色体的适应度值,然后将其进行排序
        直接在这里将网络导入进来计算适应度值
        :param:individuals 染色体
        :return:返回对颖染色体的适应度值
        """
        #step_1:提取参数,w1,b1
        new_w1,new_b1=self.reshape_weight_bias(individual)
        # print(new_w1.shape,new_b1.shape)
        train_x, train_y, test_x, test_y = rnn_data('./power_data.txt')
        # print(train_y.shape,train_x.shape)
        # Building Graphs
        config = Config(train_x,train_y)

        X = tf.placeholder(tf.float32, [None, config.timesteps, config.features])
        Y = tf.placeholder(tf.float32, [None, config.outputdims])

        # setting parameter
        epoch = config.training_epoch
        lr = config.learning_rate
        batch_size = config.minibatch_size

        # forward calcs
        pred_Y = LSTM_Network(X, config,new_w1,new_b1)
        # cost functions
        cost = tf.reduce_mean(tf.square(pred_Y - Y))
        # start session
        sess = tf.Session()
        # initial all variables
        init = tf.global_variables_initializer()
        sess.run(init)
        sum_fitness=[]

        # train_losses=[]
        # test_losses=[]

        # epoch training
        for i in range(2):
            for start, end in zip(range(0, len(train_x), batch_size),range(batch_size, len(train_x) + 1, batch_size)):
                sess.run(cost, feed_dict={X: train_x[start:end],Y: train_y[start:end]})  # 分离出batchsize个数据去迭代运算
                temp_fitness=sess.run(cost,feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                print("temp_fitness:{}".format(temp_fitness))
                sum_fitness.append(temp_fitness)

            # cost = sess.run(cost, feed_dict = {X: test_x, Y: test_y})		# 之前犯错的地方，就是因为cost操作被再次赋值，导致出错！
            # if (i + 1) % 100 == 0:
            #     train_result, loss1 = sess.run([pred_Y, cost], feed_dict={X: train_x, Y: train_y})
            #
            #     test_result, loss2 = sess.run([pred_Y, cost], feed_dict={X: test_x, Y: test_y})
            #
            #     train_losses.append(loss1)
            #     test_losses.append(loss2)
            #
            #     print('the epoch' + str(i + 1) + ': train loss = ' + '{:.9f}'.format(loss1)
            #           + ', test loss = ' + '{:.9f}'.format(loss2))
        print("========",np.sum(sum_fitness)/epoch*1.0)
        return np.sum(sum_fitness)/epoch*1.0

    def bestfitness(self,individual):
        """
        计算每一代种群的最佳适应度,以及对应的索引下标
        :return:
        """
        fitness_val=[]
        fitness_val=self.fitness_fun(individual=None)
        return min(fitness_val),np.argmin(fitness_val,axis=0)

    def avgfitness(self,individual):
        """
        计算每一代种群的平均适应度
        :return:
        """
        fitness_val=[]
        fitness_val=self.fitness_fun(individual=None)
        return np.average(fitness_val,axis=0)

    def compute_param_num(self):
        """
        返回参数之和
        :return:
        """
        w1_num=self.input_num*self.hidden_num
        b1_num=self.hidden_num
        return w1_num+b1_num,w1_num,b1_num

    #初始化编码值
    def code(self,len_chrom,bound):
        """
        染色体形式应该类似于：[12222,22222,22222,22222,222222,2222222]
        本函数将变量编码变成染色体,用于随机初始化一个种群
        :param lenchrom: 染色体长度
        :param bound: 变量的取值范围
        :return: 返回染色体的编码值,染色体长度,边界范围
        """
        #初始化一个染色体,并且要保证每一个基因变量的范围在合法范围之内。
        pick=np.random.rand(1,len_chrom.shape[1])
        #由于是浮点编码方法,所以每次生成编码的时候,要检查产生的浮点编码值是否边界范围之内
        ret=bound[0][0]+(bound[1][0]-bound[0][0])*pick
        return ret

    def test(self,bound,chrom):
        """
        用于测试所生成的浮点值是否在合理范围之类
        :param len_chrom: 染色体长度
        :param bound: 边界范围
        :param chrom: 染色体信息,既可以是种群所有染色体，也可以是单个染色体
        :return:
        """
        if chrom<=bound[0][0] and chrom>=bound[1][0]:
            return 1
        else:
            return 0
#程序的入口文件
if __name__=="__main__":
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
        "fitness":np.zeros(shape=(1,pop_size)),
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
        x=individuals["chrom"][len(individuals["chrom"])-1]
        #计算每个个体的适应度值
        individuals["fitness"][0][i]=sga.fitness_fun(individual=x)  #适应度值

    #找最好的染色体,以及对应的下标
    best_index,bestfitness=np.argmin(individuals["fitness"],axis=0),np.min(individuals["fitness"],axis=0)
    #最好的染色体
    bestchrom=individuals["chrom"][best_index]

    #迭代求解最佳的初始阀值和权值
    #这里还要改
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
            individuals["fitness"][0][j]=sga.fitness_fun(indivadual=x)
        #找到最小和最大适应度的染色体以及它们在种群中的位置
        newbest_fitness,newbest_index=np.min(individuals["fitness"],axis=0),np.argmin(individuals["fitness"],axis=0)
        worset_fitness,worest_index=np.max(individuals["fitness"],axis=0),np.argmax(individuals["fitness"],axis=0)
        #代替上一次进化中最好的染色体
        if bestfitness>newbest_fitness:
            bestfitness=newbest_fitness
            best_index=newbest_index
            bestchrom=individuals["chrom"][best_index]
        individuals["chrom"][worest_index]=bestchrom
        individuals["fitness"][0][worest_index]=bestfitness

    #最优染色体
    x=bestchrom
    #把最优初始阀值权值赋予给网络预测
    #获取最优网络权值和阀值，并且改变形状以适应神经网络训练的需求
    new_w1,new_b1=sga.reshape_weight_bias(x)
    #将最优权值保存起来,以便在LSTM神经网络下进行训练
    with open("w_b_data.txt","w") as f:
        f.write(new_w1+"\n")
        f.write(new_b1+"\n")

#step1:先用遗传算法求初始权值的最优解
#step2:利用遗传算法得到近似最优解赋值给网络权值和阀值
#step3:比较实验结果与分析




















