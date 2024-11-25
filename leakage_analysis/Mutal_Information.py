import numpy as np
from tqdm import *
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
def Mutal_information(trace_url,x_url,blocknum,types,samples):
    mi=np.zeros(samples)
    for num in trange(samples,desc="Computing Mutal information"):
        new_trace_col=[]
        new_x_col=[]
        for block in range(blocknum):
            trace=np.load(trace_url.format(block))
            x=np.load(x_url.format(block))&0xff
            new_trace_col.extend(trace[:,num])
            new_x_col.extend(x)
        t_arr=np.array(new_trace_col)
        x_arr=np.array(new_x_col)
        mi[num]=Mutal_information_core(t_arr,x_arr,types)
    plt.plot(mi)
    plt.xlabel('Sample Index')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information vs Sample Index')
    plt.show()
        
    

def Mutal_information_core(trace, x,types):
    # 将浮点数 trace 的一列 映射到整数范围内，例如乘以一个大数然后转换为整数
    num_bins = 20
    min_val=np.min(trace)
    max_val=np.max(trace)
    # trace_int = ((trace - min_val) / (max_val - min_val) * 20).astype(int)  # 假设数据在最小值和最大值之间,将其映射到0~1000整数范围内
    trace_int= np.digitize(trace,bins=np.linspace(min_val, max_val, num_bins))
    # 将trace离散到int上
    # 统计每对 (trace, x) 的出现次数
    joint_counts = np.zeros((np.max(trace_int) + 1, types), dtype=int)
    for i in range(len(trace_int)):
        joint_counts[trace_int[i], x[i]] += 1
    # 归一化得到联合概率分布
    joint_prob = joint_counts / len(x)
    
    # mi = 0
    # for i in range(joint_prob.shape[0]):
    #     for j in range(joint_prob.shape[1]):
    #         if joint_prob[i, j] > 0:
    #             mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (np.sum(joint_prob[i, :]) * np.sum(joint_prob[:, j])))
    mi =entropy(np.sum(joint_prob, axis=0)) + entropy(np.sum(joint_prob, axis=1)) - entropy(joint_prob.flatten())  
    # mi=mutual_info_score(x,trace_int)
    return mi


if __name__ =="__main__":
    trace=r""
    x=r""
    Mutal_information(trace,x,20,types=256,samples=500)
