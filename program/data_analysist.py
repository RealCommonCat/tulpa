import time
import numpy as np
from numba import cuda
import random
import math
import threading
class Data_analysist():
    def __init__(self,dependent_matrix,independent_matrix,split_quantity,memory_size,is_fast_mode,dimension,cpu_cores):
        self.cpu_cores=cpu_cores
        self.dependent_matrix=dependent_matrix
        self.independent_matrix=independent_matrix
        self.split_quantity=split_quantity
        self.memory_size=memory_size
        self.is_fast_mode=is_fast_mode
        self.dimension=dimension
        self.length=len(self.dependent_matrix)
        #数据组长度
        self.threads_size=64
        if is_fast_mode:
            self.assigned_index=random.sample(range(0,dimension),1)
            #随机确定用于分析的指定维度
            self.begin_index=self.assigned_index[0]
            self.end_index=self.begin_index+1
            print("assigned dimension: "+str(self.assigned_index))
            #[[idx,r]...]dimension
        else:
            self.begin_index=0
            self.end_index=dimension
        self.max=np.zeros([dimension,2])
        self.compute_all()
    def max_by_index(self,results,index):
        #根据指定维度索引排序
        max_=np.zeros(2)
        for i in range(len(results)):
            if results[i][index][1]>max_[1]:
                max_[0]=results[i][index][0]
                max_[1]=results[i][index][1]
        if max_[1]>self.max[index][1]:
            self.max[index][1]=max_[1]
            self.max[index][0]=max_[0]
    def max_all(self,results):
        #排序
        if self.is_fast_mode:
            self.max_by_index(results,self.assigned_index[0])
        else:
            for i in range(self.dimension):
                self.max_by_index(results,i)
    last_r=0
    def compute_splited(self,functions_space_size,results_space_size,average_independent_matrix_size,average_dependent_matrix,operated_independent_matrix_size,begin_position,section_length):
        #计算，所有的size都是函数的数量
        surplus_time=0
        begin_time=time.time()
        functions_space_=cuda.device_array(functions_space_size)
        results_space_=cuda.device_array(results_space_size)
        dependent_matrix_=cuda.to_device(self.dependent_matrix)
        independent_matrix_=cuda.to_device(self.independent_matrix)
        operated_independent_matrixs_=cuda.device_array(operated_independent_matrix_size)
        average_independent_matrixs_=cuda.device_array(average_independent_matrix_size)
        average_dependent_matrix_=cuda.to_device(average_dependent_matrix)
        gpu_compute[math.ceil(section_length/self.threads_size),self.threads_size](self.begin_index,self.end_index,functions_space_,results_space_,average_independent_matrixs_,average_dependent_matrix_,dependent_matrix_,independent_matrix_,operated_independent_matrixs_,begin_position,section_length,self.length,self.dimension,self.split_quantity)
        cuda.synchronize()
        results=results_space_.copy_to_host()
        #获取结果
        if self.is_fast_mode:
            self.max_by_index_threads(results,self.assigned_index[0])
        else:
            self.max_all_threads(results)
        end_time=time.time()
        r=(begin_position+section_length)/self.total_size
        delta_r=r-self.last_r
        self.last_r=r
        try:surplus_time=(end_time-begin_time)*(1/delta_r-1)*(1-r)
        except:
            surplus_time=0
        print("\r"+"["+"="*int(r*50)+">"+" "*(50-int(r*50))+"]    "+"%.2f%%"%(r*100)+" surplus_time:%.2f s      "%surplus_time,end="")
    def compute_all(self):
        begin=0
        self.total_size=((self.split_quantity)**self.dimension)
        #共计搜索的函数数量
        self.function_size=self.dimension**2
        #每个函数的大小 每个大小占据8byte
        block_size=self.function_size*8+(self.dimension*2)*8+self.length*self.dimension*8+self.dimension*8
        #每一个函数所需要的显存空间(byte)
        section_length=math.ceil(self.memory_size/block_size)
        #分段长度
        section_quantity=0
        #分段数量
        average_dependent_matrix=np.zeros([self.dimension])
        #提前计算好的平均值
        average_independent_matrix_size=[section_length,self.dimension]
        #平均变换后自变量矩阵占有空间    
        functions_space_size=[section_length,self.dimension,self.dimension]
        results_space_size=[section_length,self.dimension,2]
        operated_independent_matrix_size=[section_length,self.length,self.dimension]
        if block_size*self.total_size>self.memory_size:
            section_quantity=math.floor(self.total_size*block_size/self.memory_size)
        tail=self.total_size%section_length
        for i in range(self.dimension):
            for j in range(self.length):
                average_dependent_matrix[i]+=self.dependent_matrix[j][i]
            average_dependent_matrix[i]/=self.length
        if section_length>0:
            print("lack of memory and split")
        print("About to begin...")
        for i in range(section_quantity):
            self.compute_splited(functions_space_size,results_space_size,average_independent_matrix_size,average_dependent_matrix,operated_independent_matrix_size,begin,section_length)
            begin+=section_length
        self.compute_splited(functions_space_size,results_space_size,average_independent_matrix_size,average_dependent_matrix,operated_independent_matrix_size,begin,tail)
        for i in range(len(self.max)):
            print("\nR:"+str(math.sqrt(self.max[i][1])))
            function_0=self.get_function_by_index(self.max[i][0])
            print("\n"+str(function_0))
    def get_function_by_index(self,index):
        #获取指定索引的函数
        k=-1
        function_=np.zeros([self.dimension,self.dimension])
        for i in range(self.dimension):
            for j in range(self.dimension):
                k+=1
                function_[i][j]=((index/(self.split_quantity**k)%self.split_quantity)/self.split_quantity)    
        return function_
    def max_by_index_threads(self,results,index):
        #多线程排序
        flag=[0,]
        #标志位，一旦计数器达到cpu_cores，就进行排序
        section_length=math.floor(len(results)/self.cpu_cores)
        #每个核心都分到一个段落
        max_=np.zeros([self.cpu_cores,2])
        #核心记录位置
        for i in range(self.cpu_cores):
            Threads_max(results,index,i*section_length,section_length,flag,max_,i).start()
        while flag[0]<self.cpu_cores:
            pass
        for i in range(len(max_)):
            if max_[i][1]>self.max[index][1]:
                self.max[index][0]=max_[i][0]
                self.max[index][1]=max_[i][1]
    def max_all_threads(self,results):
        #排序
        if self.is_fast_mode:
            self.max_by_index_threads(results,self.assigned_index[0])
        else:
            for i in range(self.dimension):
                self.max_by_index_threads(results,i)
@cuda.jit
#144 2 高数作业
def gpu_compute(begin_index,end_index,functions_space,results_space,average_independent_matrixs,average_dependent_matrix,dependent_matrix,independent_matrix,operated_independent_matrixs,begin_position,section_length,data_length,dimension,split_quantity):
    #GPU完成一次数据的变换，输入的是标准计算矩阵，输出的是结果矩阵
    idx=cuda.threadIdx.x+cuda.blockIdx.x*cuda.blockDim.x
    index=idx+begin_position
    if idx>section_length:
        return
    function_=functions_space[idx]
    result=results_space[idx]
    average_independent_matrix=average_independent_matrixs[idx]
    operated_independent_matrix=operated_independent_matrixs[idx]
    #获得函数
    k=-1
    for i in range(dimension):
        for j in range(dimension):
            k+=1
            function_[i][j]=((index/(split_quantity**k)%split_quantity)/split_quantity)
    #线性映射
    for i in range(data_length):
        for j in range(dimension):
            operated_independent_matrix[i][j]=0
            for k in range(dimension):
                operated_independent_matrix[i][j]+=function_[j][k]*independent_matrix[i][k]
    #计算变换后平均值
    for i in range(dimension):
        average_independent_matrix[i]=0
        for k in range(data_length):
            average_independent_matrix[i]+=operated_independent_matrix[k][i]
        average_independent_matrix[i]/=data_length
    #分别计算各个维度下的相关系数，具体计算方法是将变换后的矩阵种的某一个列固定 然后和因变量的对应列进行相关计算
    for i in range(begin_index,end_index):
        r0=0
        r1=0
        r2=0  
        for j in range(data_length):
            r0+=(dependent_matrix[j][i]-average_dependent_matrix[i])*(operated_independent_matrix[j][i]-average_independent_matrix[i])
            r1+=(dependent_matrix[j][i]-average_dependent_matrix[i])**2
            r2+=(operated_independent_matrix[j][i]-average_independent_matrix[i])**2
        result[i][1]=(r0**2)/(r1*r2)
        result[i][0]=index
class Threads_max(threading.Thread):
    #并行排序
    def __init__(self,results,index,begin,section_length,flag,max_,core_index):
        threading.Thread.__init__(self)
        self.results=results
        self.index=index
        self.section_length=section_length
        self.begin=begin
        self.flag=flag
        self.max_=max_
        self.core_index=core_index
    def run(self):
        #排序
        for idx in range(self.begin,self.begin+self.section_length):
            if self.results[idx][self.index][1]>self.max_[self.core_index][1]:
                self.max_[self.core_index][0]=self.results[idx][self.index][0]
                self.max_[self.core_index][1]=self.results[idx][self.index][1]
        self.flag[0]+=1
    
        

            