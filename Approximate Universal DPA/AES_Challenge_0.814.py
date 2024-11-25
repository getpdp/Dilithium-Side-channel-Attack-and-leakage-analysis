import scipy.io as scio
## 读取数据

import numpy as np
from tqdm import *
from itertools import combinations
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
sbox = (
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
)
inv_sbox = (
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
)

def intermediate(pt, keyguess):
    return sbox[pt ^ keyguess]

HW = [bin(n).count("1") for n in range(0, 256)]

def mean(X):
    return np.mean(X, axis=0,dtype=np.float32)

def std_dev(X, X_bar):
    return np.sqrt(np.sum((X-X_bar)**2, axis=0),dtype=np.float32)

def cov(X, X_bar, Y, Y_bar):
    return np.sum((X-X_bar)*(Y-Y_bar), axis=0,dtype=np.float32)

def Mu(trace,x,types,samples):
    # 依据x的值域进行分类，对每一类的trace求均值
    mu_data=np.zeros([types])
    # data types:256, trace samples:500
    counter=np.zeros(types)
    for t in range(trace.shape[0]):
        counter[x[t]]+=1
        mu_data[x[t]]=((counter[x[t]]-1)*mu_data[x[t]]+trace[t])/counter[x[t]]
    return mu_data

def to_BinaryList(num):
    binary_str = bin(num)[2:]
    # 补齐为8位，不足的部分在前面补0
    binary_str = binary_str.zfill(8)
    # 将二进制字符串转换为比特列表
    binary_list = [int(bit) for bit in binary_str]
    return binary_list


def pre_trace(plain, mu_data):
    trace_num = plain.shape[0]
    result_trace = np.zeros([trace_num])
    for i in range(trace_num):
        result_trace[i] = mu_data[plain[i]]
        
    return result_trace

import numpy as np







def CPA_RR(trace,plain,X_matrix,mu,A,candidate):
    # 非建模版本，求解256个系数带入CPA
    maxcpa = np.zeros(256)
    bestguess = np.zeros(16, dtype=int)  # 用来存储猜测密钥

    # Initialize arrays &amp; variables to zero
    # mu_lst是依据s_val的均值矩阵
    np.set_printoptions(threshold=np.inf)
    trace=pre_trace(plain,mu)
    for kguess in range(0, 256):

        hws = np.zeros(trace.shape[0])
        for t in range(trace.shape[0]):
            s_val=sbox[plain[t]^kguess]
            vec=list(X_matrix[s_val])
            vec.extend(to_BinaryList(kguess))
            hws[t]=np.dot(vec,A[kguess])

        correlation_coefficient, p_value = pearsonr(hws, trace)
        maxcpa[kguess] = np.max(np.abs(correlation_coefficient))

    sort_maxcpa = np.sort(maxcpa)[::-1]

    for i in range(16):  # 把相关性高的前几位给 bestguess
        bestguess[i] = np.argmax(maxcpa)  # 返回最大值对应的索引（即密钥）赋给bestguess
        maxcpa[bestguess[i]] = 0

    return bestguess[:candidate],sort_maxcpa[:candidate]


def Ridge_regression(x,trace,key,X_matrix,lambda_val):
    # 5个版本依次对应sbox的输出、sbox的输入输出、p，key，sbox的输出、sbox的输出及2次项、sbox的输出及三次项、sbox的输出及4次项
    s_val=np.zeros_like(x)
    for i in range(x.shape[0]):
        s_val[i]=sbox[key^x[i]]
    # 依据sbox val进行求均值
    mu=Mu(trace,s_val,256,1)
    X=[]
    for s in range(256):
        vec = list(X_matrix[s])
        # 将二进制列表追加到s_bits
        vec.extend(to_BinaryList(key))
        X.append(vec)
    X=np.array(X)
    # 优化
    mu[0]=0
    zero_rows = np.where(mu == 0)[0]
    mu_filtered = np.delete(mu, zero_rows, axis=0)
    x_filtered = np.delete(X, zero_rows, axis=0)
    ridge_model = Ridge(alpha=lambda_val, fit_intercept=False)
    ridge_model.fit(x_filtered, mu_filtered)
    ridge_coefficients = ridge_model.coef_
    return ridge_coefficients


def RidgeR_Pre_Compute(ver):
    if ver=="ver1":
        X_matrix=np.zeros([256,8],dtype=int)
    if ver=="ver2":
        X_matrix=np.zeros([256,36],dtype=int)
    if ver=="ver3":
        X_matrix=np.zeros([256,92],dtype=int)
    if ver=="ver4":
        X_matrix=np.zeros([256,162],dtype=int)
    if ver=="ver5":
        X_matrix=np.zeros([256,218],dtype=int)

    for s_val in range(256):
        binary_list=to_BinaryList(s_val)
        if ver=="ver1":
            X_matrix[s_val]=np.array(binary_list)
        if ver=="ver2":
            product_results_2 = [binary_list[i] * binary_list[j] for i, j in combinations(range(len(binary_list)), 2)]
            binary_list.extend(product_results_2)
            X_matrix[s_val]=np.array(binary_list)
        if ver=="ver3":
            product_results_2 = [binary_list[i] * binary_list[j] for i, j in combinations(range(len(binary_list)), 2)]
            product_results_3 = [binary_list[i] * binary_list[j] * binary_list[k] for i, j, k in combinations(range(len(binary_list)), 3)]
            binary_list.extend(product_results_2)
            binary_list.extend(product_results_3)
            X_matrix[s_val]=np.array(binary_list)
        if ver=="ver4":
            product_results_2 = [binary_list[i] * binary_list[j] for i, j in combinations(range(len(binary_list)), 2)]
            product_results_3 = [binary_list[i] * binary_list[j] * binary_list[k] for i, j, k in combinations(range(len(binary_list)), 3)]
            product_results_4 = [binary_list[i] * binary_list[j] * binary_list[k] * binary_list[q] for i, j, k, q in combinations(range(len(binary_list)), 4)]
            binary_list.extend(product_results_2)
            binary_list.extend(product_results_3)
            binary_list.extend(product_results_4)
            X_matrix[s_val]=np.array(binary_list)
        if ver=="ver5":
            product_results_2 = [binary_list[i] * binary_list[j] for i, j in combinations(range(len(binary_list)), 2)]
            product_results_3 = [binary_list[i] * binary_list[j] * binary_list[k] for i, j, k in combinations(range(len(binary_list)), 3)]
            product_results_4 = [binary_list[i] * binary_list[j] * binary_list[k] * binary_list[q] for i, j, k, q in combinations(range(len(binary_list)), 4)]
            product_results_5 = [binary_list[i] * binary_list[j] * binary_list[k] * binary_list[q] * binary_list[a] for i, j, k, q, a in combinations(range(len(binary_list)), 5)]
            binary_list.extend(product_results_2)
            binary_list.extend(product_results_3)
            binary_list.extend(product_results_4)
            binary_list.extend(product_results_5)
            X_matrix[s_val]=np.array(binary_list)


    return X_matrix

            









def Select(trace,x,field_num):
    indices = list(range(len(trace)))
    # 随机选择1700个不重复的下标
    selected_indices = random.sample(indices, field_num)
    new_trace=trace[selected_indices]
    new_x=x[selected_indices]
    return new_trace, new_x, selected_indices


def Supple(trace,x,field_num,idx):
    indices = list(range(len(trace)))
    remain = list(set(indices)-set(idx))
    selected_indices = random.sample(remain, field_num)
    new_trace=trace[selected_indices]
    new_x=x[selected_indices]
    return new_trace, new_x, selected_indices





def Non_profiling_RL_find_low(trace,x,X_matrix,lambda_val):
    vote=np.zeros(256,dtype=int)
    guess=0
    key=0
    pre_t, pre_p, idx = Select(trace, x, field_num=190)
    num=pre_p.shape[0]
    indices = list(range(len(pre_t)))
    Coeff=np.zeros([256,8])
    for iter in range(30):
        selected_indices = random.sample(indices, 170)
        t=pre_t[selected_indices]
        p=pre_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
            # Coeff[kguess]=(Coeff[kguess]+A[kguess])/(iter+1)
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num

    for iter in range(10):
        selected_indices = random.sample(indices, 185)
        t=pre_t[selected_indices]
        p=pre_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num
    sup_t,sup_p,sup_idx=Supple(trace,x,field_num=80,idx=idx)
    comb_t=np.concatenate((pre_t,sup_t))
    comb_p=np.concatenate((pre_p,sup_p))
    num=comb_p.shape[0]
    indices = list(range(len(comb_t)))
    for iter in range(30):
        selected_indices = random.sample(indices, 225)
        t=comb_t[selected_indices]
        p=comb_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num

    for iter in range(30):
        selected_indices = random.sample(indices, 265)
        t=comb_t[selected_indices]
        p=comb_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num


    max_indices = np.argsort(vote)[-5:]
    if vote[max_indices[4]]>=60 and vote[max_indices[4]]+vote[max_indices[3]]<100:
        return max_indices[4],1,num
    
    vote=np.zeros(256,dtype=int)
    idx_new=idx+sup_idx
    sup_t,sup_p,sup_idx=Supple(trace,x,field_num=45,idx=idx_new)
    comb_t=np.concatenate((comb_t,sup_t))
    comb_p=np.concatenate((comb_p,sup_p))
    num=comb_p.shape[0]
    mu=Mu(comb_t,comb_p,types=256,samples=1)
    A=[]
    for kguess in range(256):
        A.append(Ridge_regression(comb_p,comb_t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
    key_list,pr = CPA_RR(comb_t, comb_p, X_matrix, mu, A, candidate=5)
    key=key_list[0]
    guess=1
    # print(key_list,pr,max_indices)
    return key,guess,num



def Non_profiling_RL_find_medium(trace,x,X_matrix,lambda_val):
    vote=np.zeros(256,dtype=int)
    guess=0
    key=0
    pre_t, pre_p, idx = Select(trace, x, field_num=390)
    num=pre_p.shape[0]
    indices = list(range(len(pre_t)))
    Coeff=np.zeros([256,8])
    for iter in range(30):
        selected_indices = random.sample(indices, 340)
        t=pre_t[selected_indices]
        p=pre_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
            # Coeff[kguess]=(Coeff[kguess]+A[kguess])/(iter+1)
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num

    for iter in range(10):
        selected_indices = random.sample(indices, 365)
        t=pre_t[selected_indices]
        p=pre_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num
    sup_t,sup_p,sup_idx=Supple(trace,x,field_num=125,idx=idx)
    comb_t=np.concatenate((pre_t,sup_t))
    comb_p=np.concatenate((pre_p,sup_p))
    num=comb_p.shape[0]
    indices = list(range(len(comb_t)))
    for iter in range(30):
        selected_indices = random.sample(indices, 400)
        t=comb_t[selected_indices]
        p=comb_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num

    for iter in range(30):
        selected_indices = random.sample(indices, 460)
        t=comb_t[selected_indices]
        p=comb_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num


    max_indices = np.argsort(vote)[-5:]
    if vote[max_indices[4]]>=60 and vote[max_indices[4]]+vote[max_indices[3]]<100:
        return max_indices[4],1,num
    
    vote=np.zeros(256,dtype=int)
    idx_new=idx+sup_idx
    sup_t,sup_p,sup_idx=Supple(trace,x,field_num=50,idx=idx_new)
    comb_t=np.concatenate((comb_t,sup_t))
    comb_p=np.concatenate((comb_p,sup_p))
    num=comb_p.shape[0]
    mu=Mu(comb_t,comb_p,types=256,samples=1)
    A=[]
    for kguess in range(256):
        A.append(Ridge_regression(comb_p,comb_t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
    key_list,pr = CPA_RR(comb_t, comb_p, X_matrix, mu, A, candidate=5)
    key=key_list[0]
    guess=1
    # print(key_list,pr,max_indices)
    return key,guess,num


def Non_profiling_RL_find_high(trace,x,X_matrix,lambda_val):
    vote=np.zeros(256,dtype=int)
    guess=0
    key=0
    pre_t, pre_p, idx = Select(trace, x, field_num=1400)
    num=pre_p.shape[0]
    indices = list(range(len(pre_t)))
    Coeff=np.zeros([256,8])
    for iter in range(30):
        selected_indices = random.sample(indices, 1000)
        t=pre_t[selected_indices]
        p=pre_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
            # Coeff[kguess]=(Coeff[kguess]+A[kguess])/(iter+1)
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num

    for iter in range(10):
        selected_indices = random.sample(indices, 1200)
        t=pre_t[selected_indices]
        p=pre_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num
    sup_t,sup_p,sup_idx=Supple(trace,x,field_num=205,idx=idx)
    comb_t=np.concatenate((pre_t,sup_t))
    comb_p=np.concatenate((pre_p,sup_p))
    num=comb_p.shape[0]
    indices = list(range(len(comb_t)))
    for iter in range(30):
        selected_indices = random.sample(indices, 1355)
        t=comb_t[selected_indices]
        p=comb_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num

    for iter in range(30):
        selected_indices = random.sample(indices, 1405)
        t=comb_t[selected_indices]
        p=comb_p[selected_indices]
        mu=Mu(t,p,types=256,samples=1)
        A = []
        for kguess in range(256):
            A.append(Ridge_regression(p,t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
        key_list,pr = CPA_RR(t, p, X_matrix, mu, A, candidate=5)
        vote[key_list[0]]+=1
        if pr[0]-pr[1]>=0.1:
            return key_list[0],1,num


    max_indices = np.argsort(vote)[-5:]
    if vote[max_indices[4]]>=60 and vote[max_indices[4]]+vote[max_indices[3]]<100:
        return max_indices[4],1,num
    
    vote=np.zeros(256,dtype=int)
    idx_new=idx+sup_idx
    sup_t,sup_p,sup_idx=Supple(trace,x,field_num=350,idx=idx_new)
    comb_t=np.concatenate((comb_t,sup_t))
    comb_p=np.concatenate((comb_p,sup_p))
    num=comb_p.shape[0]
    mu=Mu(comb_t,comb_p,types=256,samples=1)
    A=[]
    for kguess in range(256):
        A.append(Ridge_regression(comb_p,comb_t,key=kguess,X_matrix=X_matrix,lambda_val=lambda_val))
    key_list,pr = CPA_RR(comb_t, comb_p, X_matrix, mu, A, candidate=5)
    key=key_list[0]
    guess=1
    # print(key_list,pr,max_indices)
    return key,guess,num
    

def process_iteration_high_noise(trace,plain,start, end, X_matrix, lambda_val, real_key):

    trace = trace.reshape(40000)
    x = plain.reshape(40000).astype(int)
    
    guess=0
    right=0
    total_num=0

    for _ in range(start, end):
        key,g,trace_num=Non_profiling_RL_find_high(trace,x,X_matrix=X_matrix,lambda_val=lambda_val)
        guess+=g
        total_num+=trace_num
        if key == real_key:
            right += 1
    return right,total_num

def process_iteration_medium_noise(trace,plain,start, end, X_matrix, lambda_val, real_key):

    trace = trace.reshape(20000)
    x = plain.reshape(20000).astype(int)
    guess=0
    right=0
    total_num=0

    for _ in range(start, end):
        key,g,trace_num=Non_profiling_RL_find_medium(trace,x,X_matrix=X_matrix,lambda_val=lambda_val)
        guess+=g
        total_num+=trace_num
        if key == real_key:
            right += 1
    return right,total_num


def process_iteration_low_noise(trace,plain,start, end, X_matrix, lambda_val, real_key):

    trace = trace.reshape(10000)
    x = plain.reshape(10000).astype(int)

    guess=0
    right=0
    total_num=0

    for _ in range(start, end):
        key,g,trace_num=Non_profiling_RL_find_low(trace,x,X_matrix=X_matrix,lambda_val=lambda_val)
        guess+=g
        total_num+=trace_num
        if key == real_key:
            right += 1
    return right,total_num



if __name__ == '__main__':
    low_trace_url=r"Approximate Universal DPA\\leakage_low.mat"
    low_plain_url=r"Approximate Universal DPA\\plain_low.mat"
    medium_trace_url=r"Approximate Universal DPA\\leakage_medium.mat"
    medium_plain_url=r"Approximate Universal DPA\\plain_medium.mat"
    high_trace_url=r"Approximate Universal DPA\\leakage_high.mat"
    high_plain_url=r"Approximate Universal DPA\\plain_high.mat"
    plain_low = scio.loadmat(low_plain_url)['plain']
    leakage_low = scio.loadmat(low_trace_url)['leakage']
    plain_medium = scio.loadmat(medium_plain_url)['plain']
    leakage_medium = scio.loadmat(medium_trace_url)['leakage']
    plain_high = scio.loadmat(high_plain_url)['plain']
    leakage_high = scio.loadmat(high_trace_url)['leakage']
    noise_ver="high"

    if noise_ver=="low":
        ver="ver1"
        X_matrix=RidgeR_Pre_Compute("ver1")
        # r,t=process_iteration_low_noise(leakage_low,plain_low,0,1,X_matrix,2000,114)
        # print(r,t)
        #for j in range(10):
        lambda_val=2000
        num = 1000
        pnum = 10
        right = 0
        trace_num = 0
        chunk_size = num // pnum   
        key_right=114  
        with ProcessPoolExecutor(max_workers=pnum) as executor:
            futures = [executor.submit(process_iteration_low_noise, leakage_low, plain_low ,i * chunk_size, (i+1) * chunk_size if i < pnum - 1 else num,X_matrix,lambda_val,key_right) for i in range(pnum)]
            results = list(tqdm(as_completed(futures), total=pnum, desc='Collecting Results'))
            
            for future in results:
                r, part_num = future.result()
                right += r
                trace_num += part_num
        print("Success Rate:", right / num, "Average num: ", trace_num/num)

    if noise_ver=="medium":
        ver="ver1"
        X_matrix=RidgeR_Pre_Compute("ver1")
        #for j in range(10):
        lambda_val=2000
        num = 1000
        pnum = 10
        right = 0
        trace_num = 0
        chunk_size = num // pnum
        key_right=44
        with ProcessPoolExecutor(max_workers=pnum) as executor:
            futures = [executor.submit(process_iteration_medium_noise, leakage_medium, plain_medium ,i * chunk_size, (i+1) * chunk_size if i < pnum - 1 else num,X_matrix,lambda_val,key_right) for i in range(pnum)]
            results = list(tqdm(as_completed(futures), total=pnum, desc='Collecting Results'))
            
            for future in results:
                r, part_num = future.result()
                right += r
                trace_num += part_num
        print("Success Rate:", right / num, "Average num: ", trace_num/num)

    if noise_ver=="high":
        ver="ver1"
        X_matrix=RidgeR_Pre_Compute("ver1")
        #for j in range(10):
        lambda_val=2000
        num = 1000
        pnum = 10
        right = 0
        trace_num = 0
        chunk_size = num // pnum 
        key_right=211       
        with ProcessPoolExecutor(max_workers=pnum) as executor:
            futures = [executor.submit(process_iteration_high_noise, leakage_high, plain_high, i * chunk_size, (i+1) * chunk_size if i < pnum - 1 else num,X_matrix,lambda_val,key_right) for i in range(pnum)]
            results = list(tqdm(as_completed(futures), total=pnum, desc='Collecting Results'))
            
            for future in results:
                r, part_num = future.result()
                right += r
                trace_num += part_num
        print("Success Rate:", right / num, "Average num: ", trace_num/num)

        

