import numpy as np
from tqdm import *
import numba as nb
# 从全部的操作里提取operation的曲线
GAMMA=2**32

def hamming_weight(hex_string):
    # 还原Gamma1-coefficient之前的值的HW,值不会超过17
    hex_string=hex_string[6:8]+hex_string[4:6]+hex_string[2:4]+hex_string[0:2]
    if hex_string[0]=='f':
        hex_val=int(hex_string, 16)-GAMMA
    else: hex_val=int(hex_string, 16)
    origin=0x20000-hex_val
    binary_string = bin(origin)[2:]
    # 计算二进制字符串中非零位的数量
    return binary_string.count('1')


def trace_extract(operation,trace_url,data_url,position_url,block_num):
    zero_list=[]
    # 训练集里每种操作大概有5w个
    # 余下汉明重量的每种1w
    HW=[[] for _ in range(18)]
    pos=np.load(position_url.format(operation))
    # 每个操作对应的起点
    for block in trange(block_num):
        trace_file=np.load(trace_url.format(block)).astype(np.float32)
        data_file=np.load(data_url.format(block))
        coeff_data=data_file["allbytes"]
        take=np.zeros([18])
        # 每个文件每种hw提取的数量不超过18条，take是每个hw的counter
        for t in range(len(trace_file)):
            whole_trace=trace_file[t]
            for coeff in range(256):
                if coeff%4==operation:
                    round_num=coeff//4
                    coeff_val=coeff_data[t][coeff*8:coeff*8+8]
                    # 目前这个coefficient的字符串小端字符串
                    if coeff_val=="00000000":
                        zero_list.append(whole_trace[pos[round_num]:pos[round_num]+300])
                        # 认为一段操作长度为300
                    else:
                        hw_val=hamming_weight(coeff_val)
                        if len(HW[hw_val])<12000 and take[hw_val]<20:
                            HW[hw_val].append(whole_trace[pos[round_num]:pos[round_num]+300])
                            take[hw_val]+=1
            # end for coeff
        # end for t   
        del trace_file
        del data_file
    # end for block
    other_list=[]
    
    for i in range(18):
        other_list+=HW[i]
    return zero_list,other_list
                        




def Get_Param(operation,trace,other,numPOIs): # 寻找的兴趣点数量，兴趣点之间的最小间隔（防止兴趣点过于集中）
    # y_0 表示该处y_i,index==0
    # y_1 表示该处y_i,index!=0
    y_0=np.array(trace)
    y_1=np.array(other)
    y_0_means=np.average(y_0,axis=0)
    y_1_means=np.average(y_1,axis=0)
    ttest_url=r"D:\Dilithium_Paper_Work\Dilithium Paper\core code\ttest\ttest_result_coeff_{}.npy".format(operation)
    # 得到第round轮第index个系数的t-test曲线
    ttest_result=np.load(ttest_url)
    POIs = []           # 用来存放兴趣点
    POIspacing=2
    for i in range(numPOIs):
    # Find the max
        nextPOI = ttest_result.argmax() # numpy的argmax方法： 返回最大值的位置
        POIs.append(nextPOI)
    
    # Make sure we don't pick a nearby value  把选择的兴趣点左右大小为5的区间置为0，目的是去除兴趣点选择过于集中
        poiMin = max(0, nextPOI - POIspacing)           #  max(): 返回最大的值
        poiMax = min(nextPOI + POIspacing, len(ttest_result))
        for j in range(poiMin, poiMax):
            ttest_result[j] = 0
    
    Means=[np.array(y_0_means),np.array(y_1_means)]
    Traces=[np.array(y_0),np.array(y_1)] 

    return POIs,Means,Traces



def Template_Generating(operation,trace,other,numPOIs ):
    POIs,Means,Traces=Get_Param(operation,trace,other,numPOIs )
    # 得到第round轮第index个操作的POI
    meanMatrix = np.zeros((2, numPOIs))     # y的均值向量
    covMatrix  = np.zeros((2, numPOIs, numPOIs))
    # print("Generating Mean Matrix,Cov Matrix for the template......")
    # coeff代表该系数是否为0
    for coeff in range(2):
        for i in range(numPOIs):
            meanMatrix[coeff][i]=Means[coeff][POIs[i]]
            for j in range(numPOIs):
                a=Traces[coeff][:,POIs[i]]
                b=Traces[coeff][:,POIs[j]]
                
                covMatrix[coeff,i,j]=cov(a,b)
    

    return meanMatrix,covMatrix,POIs


def cov(x, y):
    # Find the covariance between two 1D lists (x and y).
    # Note that var(x) = cov(x, x)
    return np.cov(x, y)[0][1]  



if __name__ == '__main__':
    # trace,others=data_profiling_Trace_integration(50)
    import gc
    trace_url=r"D:\Dilithium_Paper_Work\Dilithium_paper_traces\Dilithium_paper_profile\traces_for_profiling_part{}.npy"
    data_url=r"D:\Dilithium_Paper_Work\Dilithium_paper_metadata\metadata_files\profiling_metadata\metadata_profiling_part{}.npz"
    position_url=r"F:\Dilithium_Paper_Work\Dilithium Paper\core code\ttest\Coeff_{}.npy"
    print("\n\n Generating Mean Matrix,Cov Matrix for the template......")
    for index in trange(4):
        trace,others=trace_extract(operation=index,trace_url=trace_url,data_url=data_url,position_url=position_url,block_num=560)
        savepath=r"F:\Dilithium_Paper_Work\template\template_combined\75_pois\template_combined_75_coeff{}".format(index)
        meanMatrix,covMatrix,POIs=Template_Generating(index,trace,others,numPOIs=75)
        np.savez_compressed(savepath,mean=meanMatrix,cov=covMatrix,poi=POIs)
        del trace
        del others
    gc.collect()