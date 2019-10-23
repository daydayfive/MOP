# -*- coding: UTF-8 -*-
#@author Wu Junjun
#create time:2019/10/23
#test the moea/d(Tchebycheff Approach)


import numpy as np
import random
import math
import matplotlib.pyplot as plt
class Moea_d(object):
    def __init__(self,benchmark,H,T,V,M,Gen):
        self.benchmark=benchmark            #测试函数的选择
        self.H=H                            #用户自定义正整数，根据该正整数及问题维度可确定种群数目
        self.T=T                            #领域中权重向量的个数
        self.V=V                            #自变量个数
        self.M=M                            #问题维度
        self.Gen=Gen                        #迭代次数

    def Population_Init(self,popsize,V):              #种群的初始化
        if self.benchmark=="ZDT1":
            lb=np.zeros((popsize,V))
            ub=np.ones((popsize,V))
            x=lb+(ub-lb)*np.random.rand(popsize,V)

        return x

    def Weight_Init(self,popsize,T):                  #权重的初始化
        self.weights=np.zeros((popsize,self.M))
        w1=np.linspace(0,1,popsize)
        w2=1-w1
        self.weights[:,0]=w1
        self.weights[:,1]=w2
        self.weights[np.argwhere(self.weights==0)]=0.00001
        dis=np.zeros((popsize,popsize))
        neighbor=[]
        for i in range(popsize):
            dis[i,:]=np.linalg.norm(self.weights[i]-self.weights,ord=2,axis=1)
            dislist=list(np.argsort(dis[i,:]))
            dislist.remove(i)
            neighbor.append(dislist[:self.T])

        self.neighbor=neighbor
        # self.neighbor=np.array(neighbor)
        # self.neighbor[np.argwhere(self.neighbor==0)]=0.001
        # self.neighbor=list(self.neighbor)



    def factorial(self,N):                            #阶乘函数
        fac=1
        for i in range(1,N+1):
            fac=fac*i
        return fac

    def evalute(self,x):
        if self.benchmark=="ZDT1":
            F1=x[:,0].reshape(-1,1)

            g=(1+9*(np.sum(x,axis=1)-x[:,0])/(self.V-1)).reshape(-1,1)

            F2=g*(1-np.sqrt(F1/g))

            f=np.concatenate((F1,F2),axis=1)
            #print([F1,F2])

            return f

    def evolution(self,x,j):                              #进化算法,选用差分进化算法
        CR=0.5                                            #变异率
        F=0.5                                             #缩放因子
        neighbor=self.neighbor[j]
        s=random.sample(neighbor,3)                       #随机选取3个索引

        p1=x[s[0]]
        p2=x[s[1]]
        p3=x[s[2]]                                        #j!=s0!=s1!=s2!=s3

        #变异操作

        vj=p1+F*(p2-p3)
        vj=self.fixnew(vj)

        #交叉操作

        jrand=np.random.randint(self.V)

        select=np.random.rand(self.V)<=CR

        select[jrand]=True

        newp=x[j]
        newp[select]=vj[select]

        #选择操作
        return newp


    def fixnew(self,x):                                 #修复超出范围的个体
        x[x<0]=0
        x[x>1]=1
        return x

    def updates(self,newp,z,k):                                #更新邻域解
        weight=self.weights[k]

        gtey_=np.max(weight*np.fabs(z-self.z))

        gtex=np.max(weight*np.fabs(self.y[k]-self.z))


        if gtey_<gtex:
            self.x[k]=newp
            self.y[k]=z



    def Run(self):
        popsize=int(self.factorial(self.H+self.M-1)/(self.factorial(self.M-1)*self.factorial(self.H)))

        self.x=self.Population_Init(popsize,self.V)
        self.y=np.array(self.evalute(self.x))
        #print(self.y)
        self.Weight_Init(popsize,self.M)

        self.z=np.min(self.y,axis=0).reshape(1,-1)



        for i in range(self.Gen):
            for j in range(popsize):
                newp=self.evolution(self.x,j).reshape(1,-1)
                newp=self.fixnew(newp)

                z=np.array(self.evalute(newp))


                self.z[z<self.z]=z[z<self.z]

                for k in self.neighbor[j]:

                    self.updates(newp,z,k)


    def plot_f(self):
        print(self.y[-1,0],self.y[-1,1])
        plt.scatter(self.y[:,0],self.y[:,1])
        plt.show()




if __name__=="__main__":
    model=Moea_d("ZDT1",100,20,30,2,200)
    model.Run()
    model.plot_f()



