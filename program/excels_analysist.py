from threading import Thread
import threading
import pandas
import numpy as np
import os
import data_analysist
class Excels_analysist():
    #将EXCELS格式的文件读入
    split_quantity=0
    #分割数量，也就是分割数量
    memory_size=0
    #准备分配的总内存的大小，单位为字节
    excels_path=0
    #excels文件的路径
    is_fast_mode=False
    #是否是快速模式
    dimension=0
    #向量维度

    def excels_analyze(self):
        print("Welcome to use the data analysist R_ made by common-cat             mew mew =w=")
        self.excels_path=input("please enter the excels path:\n")
        self.memory_size=int(input("please enter the gpu memory size(GB):\n"))
        self.split_quantity=int(input("please enter the split quantity(suggestion:1000 || bigger better):\n"))
        self.cpu_cores=int(input("please enter the cpu cores quantity:\n"))
        if input("fast mode?(y/n)\n")=="y":
            self.is_fast_mode=True
        #将excels文件读入并变换成需要的形式
        data=self.get_standard_matrix(self.remove_not_in_every_files_hosts(self.arrange_tulpes(self.get_excels_matrix(self.excels_path))))
        data_analysist.Data_analysist(data[1],data[0],self.split_quantity,self.memory_size*(2**29),self.is_fast_mode,self.dimension,self.cpu_cores)
    def get_excels_matrix(self,path):
        #获取excels文件的矩阵,返回分析的向量维度（一个文件一套指标一个向量维度）
        #数据格式为第一列为宿主名，第二列为宿主value，TULPA紧跟其后
        i=0
        #计数器，代表每个维数的含义
        files=os.listdir(path)
        excels_matrix=[]
        #excels文件的矩阵
        for file in files:
            try:pandas.read_excel(path+"/"+file)
            except:continue
            excels_matrix.append(pandas.read_excel(path+"/"+file).values)
            print(str(i)+":"+str(file))
            i+=1
        self.dimension=len(excels_matrix)
        return excels_matrix
    def arrange_tulpes(self,excels_matrix):
        #将文件的内容组合为元组,其格式为dimension->dic[host_name->list[[h,t]...]]
        excels_matrix_tulpes=[]
        #excels文件的矩阵的元组
        for i in range(self.dimension):
            host_tulpas_dic={}
            #宿主TULPA对
            for j in range(len(excels_matrix[i])):
                host_tulpas_dic[excels_matrix[i][j][0]]=[]
                for k in range(1,len(excels_matrix[i][j])):
                    host_tulpas_dic[excels_matrix[i][j][0]].append([excels_matrix[i][j][1],excels_matrix[i][j][k]]) 
            excels_matrix_tulpes.append(host_tulpas_dic)
        return excels_matrix_tulpes
    def remove_not_in_every_files_hosts(self,excels_matrix_tulpes):
        #删除不在所有文件内的宿主,数据结构同上
        del_hosts=[]
        #即将被删除的宿主
        for excels_matrix_tulpe in excels_matrix_tulpes:
            hosts=list(excels_matrix_tulpe.keys())
            for host in hosts:
                for excels_matrix_tulpe_ in excels_matrix_tulpes:
                    if host not in list(excels_matrix_tulpe_.keys()):
                        del_hosts.append(host)
        for del_host in del_hosts:
            for excels_matrix_tulpe in excels_matrix_tulpes:
                try:del excels_matrix_tulpe[del_host]
                except:pass
        return excels_matrix_tulpes
    def get_standard_matrix(self,excels_matrix_tulpes):
        #获取可以用于计算的标准矩阵，其分为自变量矩阵和因变量矩阵，其列数为样本数，也就是TULPA的数量，其行数为维度的数量
        independent_matrix=[]
        dependent_matrix=[]
        i=0
        #起点
        j=0
        #数量
        #分别为自变量矩阵和因变量矩阵
        #提取方法为相同的宿主按照维度排序，然后将前一项放入自变量矩阵相应位置，后一项则是因变量矩阵相应位置
        for host_name in excels_matrix_tulpes[0]:
            for k in range(len(excels_matrix_tulpes[0][host_name])):
                independent_matrix.append([])
                dependent_matrix.append([])
                j+=1
            #初始化一个宿主所有的TULPA的数据对
            m=0
            for k in range(i,j):
                for excels_matrix_tulpe in excels_matrix_tulpes:
                    independent_matrix[k].append(excels_matrix_tulpe[host_name][m][0])
                    dependent_matrix[k].append(excels_matrix_tulpe[host_name][m][1])
                m+=1    
            i=j           
        return [independent_matrix,dependent_matrix]