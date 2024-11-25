import numpy as np
from tqdm import *
import matplotlib.pyplot as plt
import gc
Gamma=2**32
def hamming_weight(hex_string):
    # 将整数转换为二进制字符串，并去掉开头的 '0b'
    hex_string=hex_string[6:8]+hex_string[4:6]+hex_string[2:4]+hex_string[0:2]
    if hex_string[0]=='f':
        val=int(hex_string,16)-Gamma
    else:val=int(hex_string,16)
    re=0x2000-val
    binary_string = bin(re)[2:]
    # 计算二进制字符串中 '1' 的数量，即汉明重量
    hamming_weight = binary_string.count('1')
    # if hamming_weight==0 or hamming_weight>15:
    #     print(hamming_weight)
    return hamming_weight

    

def Extract_traces(trace_url,blocksize,blocknum):
    Extract_trace=[]
    trace_example=np.load(trace_url.format(0))

    for i in trange(blocknum):
        trace=np.load(trace_url.format(i))
        for j in range(blocksize):
            x=trace[j]
            Extract_trace.append(np.array(x))
    return Extract_trace



def Extract_Coeff(data_url,round,index,blocknum,blocksize):
    Extract_data=[]
    for i in range(blocknum):
        coeff_data=np.load(data_url.format(i))
        Coeff=coeff_data["allbytes"]
        for j in range(blocksize):
            coeff_index=round*4+index
            coeff_val=Coeff[j][coeff_index*8:coeff_index*8+8]
            
            # round有64轮，每轮4个index
            Extract_data.append(hamming_weight(coeff_val))
    return Extract_data



if __name__=="__main__":
    trace_url=r"D:\Dilithium_Paper_Work\Dilithium_paper_traces\Dilithium_paper_profile\traces_for_profiling_part{}.npy"
    coeff_url=r'D:\Dilithium_Paper_Work\Dilithium_paper_metadata\metadata_files\profiling_metadata\metadata_profiling_part{}.npz'
    index_list=[0,1,2,3]
    Coeff_0=[]
    Coeff_1=[]
    Coeff_2=[]
    Coeff_3=[]
    gc.collect()
    trace_example=np.load(trace_url.format(0))
    traces=Extract_traces(trace_url=trace_url,blocksize=10,blocknum=20)
    for round in trange(1,desc="Itrerting Rounds"):
        
        N=traces[0].shape[0]
        for index in index_list:
            Coeff=Extract_Coeff(data_url=coeff_url,round=round,index=index,blocknum=20,blocksize=500)
            
            trace_arr=np.array(traces)
            coeff_arr=np.array(Coeff)
            result=np.zeros([N])
            for i in range(N):
                result[i]=np.abs(np.corrcoef(coeff_arr,trace_arr[:,i])[0,1])
                '''寻找开始位置'''
                if index==0 and result[i]>0.065:
                    if round==0:
                        Coeff_0.append(i)
                        break
                    if round>0 and i-Coeff_0[round-1]>680:
                        Coeff_0.append(i)
                        break
                if index==1 and result[i]>0.065:
                    if round==0:
                        Coeff_1.append(i)
                        break
                    if round>0 and i-Coeff_1[round-1]>680:
                        Coeff_1.append(i)
                        break
                if index==2 and result[i]>0.07:
                    if round==0:
                        Coeff_2.append(i)
                        break
                    if round>0 and i-Coeff_2[round-1]>680:
                        Coeff_2.append(i)
                        break
                if index==3 and result[i]>0.066:
                    if round==0:
                        Coeff_3.append(i)
                        break
                    if round>0 and i-Coeff_3[round-1]>680:
                        Coeff_3.append(i)
                        break
                del Coeff
            plt.plot(result)
    plt.xlabel("Samples",fontproperties = 'Times New Roman')
    plt.ylabel("Correlation Value",fontproperties = 'Times New Roman')
    plt.show()
        
    c0=np.array(Coeff_0)
    np.save(r"C:\Users\DELL\Desktop\start\Coeff_0.npy",c0)
    c1=np.array(Coeff_1)
    np.save(r"C:\Users\DELL\Desktop\start\Coeff_1.npy",c1)
    c2=np.array(Coeff_2)
    np.save(r"C:\Users\DELL\Desktop\start\Coeff_2.npy",c2)
    c3=np.array(Coeff_3)
    np.save(r"C:\Users\DELL\Desktop\start\Coeff_3.npy",c3)
    print(c0)
    print(c1)
    print(c2)
    print(c3)
    
    
        

        
    
        



