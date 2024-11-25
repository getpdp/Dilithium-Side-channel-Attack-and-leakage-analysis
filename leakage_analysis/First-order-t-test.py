import numpy as np
from tqdm import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numba import prange
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numba as nb
import random



def calculate_first_order_ttest(field_num, n, file_url, label_url, rows, cols, version, result):

    count = 0
    n_even = 0
    n_odd = 0
    mean_even = np.zeros(cols)
    M2_even = np.zeros(cols)

    mean_odd = np.zeros(cols)
    M2_odd = np.zeros(cols)


    indices=range(n)
    for p in trange(n, desc="Step1:"):
        part_select=indices[p]
        trace = np.load(file_url.format(part_select))
        label = np.load(label_url.format(part_select))
        n_even, mean_even, M2_even, n_odd, mean_odd, M2_odd, count = calculate_first_order_ttest_core(trace,label,n_even, mean_even, 
                                                                                                      M2_even, n_odd, mean_odd, M2_odd, count, rows, cols, version, result)
    var_even = M2_even / (n_even - 1)  # 无偏方差
    var_odd = M2_odd / (n_odd - 1) # 无偏方差
    temp1_1order = mean_even - mean_odd
    temp2_1order = (var_even / n_even) + (var_odd / n_odd)
    test_result_1order = temp1_1order / np.sqrt(temp2_1order)
    return test_result_1order



def iter_process(start, end, file_url, label_url, rows, cols, version):
    count = 0
    n_even = 0
    n_odd = 0
    mean_even = np.zeros(cols)
    M2_even = np.zeros(cols)

    mean_odd = np.zeros(cols)
    M2_odd = np.zeros(cols)

    result=[]
    for p in trange(start, end):
        trace = np.load(file_url.format(p))
        label = np.load(label_url.format(p))
        n_even, mean_even, M2_even, n_odd, mean_odd, M2_odd, count = calculate_first_order_ttest_core(trace,label,n_even, mean_even, 
                                                                                                      M2_even, n_odd, mean_odd, M2_odd, count, rows, cols, version, result)

    return n_even, mean_even, M2_even, n_odd, mean_odd, M2_odd, count



def concurrency_ttest(n, file_url, label_url, rows, cols, version):

    pnum = 10
    chunk_size = n // pnum
    n_even_all=0
    n_odd_all=0
    mean_even_all = np.zeros(cols)
    mean_odd_all = np.zeros(cols)
    M2_even_all = np.zeros(cols)
    M2_odd_all = np.zeros(cols)
    with ProcessPoolExecutor(max_workers=pnum) as executor:
        futures = [executor.submit(iter_process, i * chunk_size, (i+1) * chunk_size if i < pnum - 1 
                                    else n, file_url, label_url, rows, cols, version) for i in range(pnum)]
        results = list(tqdm(as_completed(futures), total=pnum, desc='Collecting Results'))
        for future in results:
            n_even, mean_even, M2_even, n_odd, mean_odd, M2_odd, count = future.result()

            M2_even_all = M2_even+M2_even_all+(mean_even-mean_even_all)**2*(n_even*n_even_all)/(n_even+n_even_all)
            M2_odd_all = M2_odd+M2_odd_all+(mean_odd-mean_odd_all)**2*(n_odd*n_odd_all)/(n_odd+n_odd_all)

            mean_even_all=(mean_even*n_even+mean_even_all*n_even_all)/(n_even+n_even_all)

            mean_odd_all=(mean_odd*n_odd+mean_odd_all*n_odd_all)/(n_odd+n_odd_all)

            

            n_even_all+=n_even
            n_odd_all+=n_odd

    mean_even = mean_even_all
    mean_odd = mean_odd_all

    var_even = M2_even_all / (n_even_all - 1)  # 无偏方差
    var_odd = M2_odd_all / (n_odd_all - 1) # 无偏方差
    temp1_1order = mean_even - mean_odd
    temp2_1order = (var_even / n_even_all) + (var_odd / n_odd_all)
    test_result_1order = temp1_1order / np.sqrt(temp2_1order)
    return test_result_1order


    


def calculate_first_order_ttest_core(trace,label,n_even, mean_even, M2_even,n_odd, mean_odd, M2_odd, count, rows, cols, version, result):

    for i_th in range(rows):

        if version=="ver1":
            # fix-vs-random is determined by odd and even.
            if count % 2 == 0:
                n1_even = n_even
                n_even = n_even + 1
                delta_even = trace[i_th] - mean_even
                delta_n_even = delta_even / n_even
                term1_even = delta_even * delta_n_even * n1_even
                mean_even = mean_even + delta_n_even

                M2_even = M2_even + term1_even
                count += 1

            else:
                n1_odd = n_odd
                n_odd = n_odd + 1
                delta_odd = trace[i_th] - mean_odd
                delta_n_odd = delta_odd / n_odd
                term1_odd = delta_odd * delta_n_odd * n1_odd
                mean_odd = mean_odd + delta_n_odd
                M2_odd = M2_odd + term1_odd
                count += 1
        if version=="ver2":
            # fix-vs-random is determined by a label value.
            if label[i_th] == 0:
                n1_even = n_even
                n_even = n_even + 1
                delta_even = trace[i_th] - mean_even
                delta_n_even = delta_even / n_even
                term1_even = delta_even * delta_n_even * n1_even
                mean_even = mean_even + delta_n_even
                M2_even = M2_even + term1_even
                count += 1

            else:
                n1_odd = n_odd
                n_odd = n_odd + 1
                delta_odd = trace[i_th] - mean_odd
                delta_n_odd = delta_odd / n_odd
                term1_odd = delta_odd * delta_n_odd * n1_odd
                mean_odd = mean_odd + delta_n_odd
                M2_odd = M2_odd + term1_odd
                count += 1
        if count%100==0:
            var_even = M2_even / (n_even - 1)  # 无偏方差
            var_odd = M2_odd / (n_odd - 1) # 无偏方差
            temp1_1order = mean_even - mean_odd
            temp2_1order = (var_even / n_even) + (var_odd / n_odd)
            test_result_1order = temp1_1order / np.sqrt(temp2_1order)
            max_val=np.max(np.abs(test_result_1order[800:]))
            result.append(max_val)

    return n_even, mean_even, M2_even,n_odd, mean_odd, M2_odd, count

# 验证了是对的
if __name__ == '__main__':

    # 文件块数
    batches = 1400
    field_num = batches
    # 路径





    label_url=r""
    file_url=r""
    save_url=r""


    first_part_trace = np.load(file_url.format(0))
    label=np.load(label_url.format(0))
    # print(label)
    traces_per_part=first_part_trace.shape[0]/10000
    rows = first_part_trace.shape[0]
    cols = first_part_trace.shape[1]
    

    # 
    # 1st ttest
    # '''Original'''
    result=[]
    first_order_ttest_result = calculate_first_order_ttest(field_num, batches, file_url, label_url, rows, cols, version="ver2", result=result)
    result=np.array(result)
    print((result))
    np.save(save_url.format("inter"),result)


    '''Concurrency'''
    # first_order_ttest_result = concurrency_ttest(field_num, file_url, label_url, rows, cols, version="ver2")
    # print(first_order_ttest_result)
    # np.save(save_url.format(int(batches*traces_per_part)),first_order_ttest_result)
    np.save(save_url.format("1st"),first_order_ttest_result)
    # 用这个2023-12-29
    plt.rcParams['figure.figsize'] = (12.0, 7.0)
    f, ax = plt.subplots(1, 1, dpi=150)
    line_value=4.5
    ax.axhline(y=line_value, ls='--', c='red', linewidth=2)
    ax.axhline(y=-1*line_value, ls='--', c='red', linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=14)  # 修改刻度的字体大小
    ax.set_yticks([-1*line_value, -4, -2, 0, 2, 4, line_value])
    # ax.set_yticks([-30, -20, -10, -1*line_value, 0, line_value, 10, 20, 30])
    ax.set_xlabel("Samples",fontproperties = 'Times New Roman', size = 15)
    ax.set_ylabel("t-statistics",fontproperties = 'Times New Roman', size = 15)
    plt.plot(first_order_ttest_result[800:])
    plt.show()

