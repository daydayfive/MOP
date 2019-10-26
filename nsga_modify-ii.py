# -*- coding: UTF-8 -*-
# @author Wu Junjun
#create time 2019/10/24
#nsga-ii 对ga交叉跟变异做了一点修改，令变异是基于子代的

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

class NSGA_II(object):
    def __init__(self,benchmark,popsize,V,M,Gen):
        self.benchmark=benchmark                           #测试基准函数
        self.popsize=popsize                               #种群大小
        self.V=V                                           #变量维度
        self.M=M                                           #问题维度
        self.Gen=Gen                                       #迭代次数

    def Population_Init(self):                             #种群初始化
        if self.benchmark=="ZDT1" or self.benchmark=="ZDT2":
            lb=np.zeros((self.popsize,self.V))
            ub=np.ones((self.popsize,self.V))
            x=lb+np.random.rand(self.popsize,self.V)*(ub-lb)
        return x

    def sorted_values(self,front,fm):                     #进行排序
        #print(fm)
        index=np.argsort(fm)
       # print(index)#进行升序
        front=np.array(front)
        real_index=front[index]
        return real_index

    def evalute(self,x):                                   #评估x
        if self.benchmark=="ZDT1":
            F1=x[:,0].reshape(-1,1)

            g = (1 + 9 * (np.sum(x, axis=1) - x[:, 0]) / (self.V - 1)).reshape(-1, 1)

            F2 = g * (1 - np.sqrt(F1 / g))

            f = np.concatenate((F1, F2), axis=1)

            return f

    def fix_new(self,x):
        x[x<0]=0
        x[x>1]=1
        return x

    def fast_non_dominated_sorted(self,x,y):                           #快速非支配排序
        num=x.shape[0]
        #print(num)
        S=[[] for i in range(num)]                            #x[i]支配的集
        n=[0 for i in range(num)]                             #x[i]被支配的数量
        rank=[0 for i in range(num)]                          #属于第几个Pareto解集
        front=[[]]
        for i in range(x.shape[0]):

            for j in range(x.shape[0]):

                dominated1=np.sum(y[i]<y[j])
                dominated2=np.sum(y[i]<=y[j])
                # print(y[i],y[j])
                # print(dominated1,dominated2)
                # print(n[i],S[i])
                if dominated1>0 and dominated2==len(y[j]):              #满足条件则x[i]支配x[j]
                    S[i].append(j)
                elif dominated1 ==0 and dominated2 <len(y[j]):
                    n[i]+=1

            if n[i]==0:
                rank[i]=0
                front[0].append(i)

        #print(front[0])
        i=0
        #initialize the front counter
        while (front[i]!=[]):
            Q=[]
            for p in front[i]:
                for q in S[p]:
                    n[q]=n[q]-1
                    if n[q]==0:
                        rank[q]=i+1                                    #当支配q的解只有一个时，因此位于下一等级的帕累托解集
                        if q not in Q:
                            Q.append(q)
            i+=1
            front.append(Q)


        del front[-1]
        return rank,front

    def crowding_distance_assignment(self,x,y,front):                 #拥挤度计算

        ssorted=[[]for i in range(len(front))]
        for i in range(len(front)):

            for j in range(self.M):                        #对每个目标函数
                ssorted[i].append(self.sorted_values(front[i],y[front[i],j]))


        #开始计算拥挤度
        n=[0 for i in range(len(x))]                                                      #nd=0

        for i in range(len(front)):
            for j in range(self.M):
                for k in range(len(front[i])):
                    if k==0 or k==len(front[i])-1:
                        n[ssorted[i][j][k]]=math.inf
                    else:
                        n[ssorted[i][j][k]]+=(y[ssorted[i][j][k+1],j]-y[ssorted[i][j][k-1],j])/(np.max(y[front[i],j])-np.min(y[front[i],j]))



        return n



    def tournament_selection(self,father,rank,crowed,pool_size,tour_size):                             #锦标赛选择算法，就是从父代种群中随机选择n个个体，选择其中最优的个体放入子代种群，知道子代种群数目达到种群要求大小
        suiji_pool=[i for i in range(self.popsize)]
        #每次随机选个2个个体，优先选择排序等级高的个体，如果排序等级一样，优先选择拥挤度大的个体
        spring_off=np.zeros([self.popsize,self.V])
        for i in range(pool_size):
            canidate=random.sample(suiji_pool,tour_size)

            if rank[canidate[0]]<rank[canidate[1]]:            #如果排序等级高，则选择它
                spring_off[i,:]=father[canidate[0],:]
            elif rank[canidate[0]]==rank[canidate[1]]:
                if crowed[canidate[0]]>crowed[canidate[1]]:
                    spring_off[i,:]=father[canidate[0],:]
                else:
                    spring_off[i,:]=father[canidate[1],:]
            else:
                spring_off[i,:]=father[canidate[1],:]


        return spring_off


    def GA(self,father):                     #遗传算法采用二进制交叉（SBX）与多项式变异(Polynomial)
        #采用每次选择2个父代进行交叉产生子代后再根据概率变异，每次生成2个体
        et=20
        et2=20
        rran=[i for i in range(self.popsize)]
        #二进制交叉
        index=0
        spring_off=np.zeros((self.popsize,self.V))
        Pc=0.9
        Pm=0.1

        # crossover=0
        # mutation=0
        while(index<self.popsize):
            child = random.sample(rran, 2)
            parent1 = father[child[0], :].reshape(1, -1)
            parent2 = father[child[1], :].reshape(1, -1)

            if random.random()<Pc:
                u = np.random.rand(2, self.V)
                gama = np.zeros((2, self.V))

                gama[u < 0.5] = np.power(2 * u[u < 0.5], 1 / (et + 1))
                gama[u >= 0.5] = np.power(1 / (2 * (1 - u[u >= 0.5])), 1 / (et + 1))

                child1 = 0.5 * ((1 + gama[0]) * parent1 + (1 - gama[0]) * parent2)
                child2 = 0.5 * ((1 - gama[1]) * parent1 + (1 + gama[1]) * parent2)
                crossover = 1
               # mutation = 0
            else:
                child1=parent1
                child2=parent2
               # crossover=0
              #  mutation=0

            if random.random()<Pm:
                # 多项式变异
                #parent3 = father[random.randint(0,self.popsize-1), :].reshape(1, -1)
                u2 = np.random.rand(2, self.V)
                delta = np.zeros((2, self.V))

                delta[u2 < 0.5] = np.power(2 * u2[u2 < 0.5], 1 / (et2 + 1)) - 1
                delta[u2 >= 0.5] = 1 - np.power(2 * (1 - u2[u2 >= 0.5]), 1 / (et2 + 1))

                child3 = child1 + delta[0]
                child4 = child2 + delta[1]
                #crossover=0
               # mutation=1
            else:
                child3=child1
                child4=child2

            if index+1<=self.popsize-1:
                spring_off[index,:]=child3
                spring_off[index+1,:]=child4
            else:
                spring_off[index,:]=child3

            index+=2

            # if crossover==1 and mutation==1:
            #
            #     if index + 1 <= self.popsize-1:
            #         spring_off[index, :] = child1
            #         spring_off[index + 1, :] = child2
            #     else:
            #         spring_off[index, :] = child1
            #     index += 2
            #     crossover=0
            #
            # elif mutation==1:
            #     spring_off[index,:]=child3
            #     index+=1
            #     mutation=0



        spring_off=self.fix_new(spring_off)
        return spring_off

    def elitism(self,father,son):             #精英主义对父代和子代合并后选择出N个种群
        new_population=np.concatenate((father,son),axis=0)
        new_value=self.evalute(new_population)

        rank,front=self.fast_non_dominated_sorted(new_population,new_value)

        select_num=0
        new_father=np.zeros((self.popsize,self.V))
        for i in range(len(front)):
            if select_num+len(front[i])<=self.popsize:
                new_father[select_num:select_num+len(front[i]),:]=new_population[front[i]]
                select_num+=len(front[i])

            else:

                rest_front=[front[i]]

                crow=self.crowding_distance_assignment(new_population,new_value,rest_front)                                        #通过对最后该等级的帕累托集进行拥挤距离计算，选择最优的个体加入它
                rest=new_population[np.argsort(crow)][::-1]                                                                        #降序排序

                new_father[select_num:]=rest[:self.popsize-select_num]
                break

        return new_father






    def Run(self):
        self.father=self.Population_Init()
        self.father_y=self.evalute(self.father)

        for i in range(self.Gen):
                # plt.scatter(self.father_y[:, 0], self.father_y[:, 1], c="b")
                # plt.show()
                rank,father_front=self.fast_non_dominated_sorted(self.father,self.father_y)
                self.crowd=self.crowding_distance_assignment(self.father,self.father_y,father_front)
                self.father=self.tournament_selection(self.father,rank,self.crowd,self.popsize,2)                          #进行锦标赛选择产生下一代的父代
                self.spring_off=self.GA(self.father)

                self.father=self.elitism(self.father,self.spring_off)
                self.father_y=self.evalute(self.father)
                # plt.scatter(self.father_y[:, 0], self.father_y[:, 1], c="b")
                # plt.show()



    def plot_f(self):
        if self.benchmark == "ZDT1":
            zdt1_pf = pd.read_table("../PF/ZDT1.txt", header=None, sep="    ")
            plt.scatter(zdt1_pf[0], zdt1_pf[1], c="r")
           # print(self.father)
            print(self.father_y)
            plt.scatter(self.father_y[:, 0], self.father_y[:, 1], c="b")
            plt.show()
        elif self.benchmark == "ZDT2":
            zdt2_pf = pd.read_table("../PF/ZDT2.txt", header=None, sep="    ")
            plt.scatter(zdt2_pf[0], zdt2_pf[1], c="r")
            plt.scatter(self.father_y[:, 0], self.father_y[:, 1], c="b")
            plt.show()


if __name__=="__main__":
    model=NSGA_II("ZDT2",100,30,2,2000)
    model.Run()
    model.plot_f()