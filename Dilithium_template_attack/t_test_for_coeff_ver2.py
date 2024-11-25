import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import time
import numba as nb
from tqdm import trange




def t_test(n, url_trace,data_url,coeff_index):
    '''
    :instrctions:even  and  odd 间或采集
    :param n: blocks of traces
    :param N: sample numbers
    :return: result
    '''
    arr = np.load(url_trace.format(block_num_start))
    count = 0
    N = 300
    
    bias=0
    old_var_y_0 = np.zeros(N-bias)
    old_mean_y_0 = np.zeros(N-bias)
    old_var_y_1 = np.zeros(N-bias)
    old_mean_y_1 = np.zeros(N-bias)
    y_1count = 0
    y_0count = 0
    pos=np.load(r"C:\Users\DELL\Desktop\start\Coeff_{}.npy".format(coeff_index))

    for j in trange(n):
        arr = np.load(url_trace.format(block_num_start+j))
        data = np.load(data_url.format(block_num_start+j))

        coeff=data['allbytes']

        for i in range(arr.shape[0]):
            # i是arr中曲线的数目
            for loop in range(64):
                # loop是64轮的轮数，coeff_index是第几个操作
                
                
                current_round=4*loop+coeff_index
                # 当前是256个操作中的第几个
                start_samples=pos[loop]
                end_samples=pos[loop]+N
                # 找到切片中对于的起止点

                trace=arr[i][start_samples:end_samples]

                
                if coeff[i][current_round*8:current_round*8+8] == '00000000':
                    # coeff长2048，每一轮长32，每一个coeff_index长8
                    # 找到第loop轮中第coeff_index为0的系数对应的曲线进行t-test
                    new_mean_y_0 = old_mean_y_0 + (trace - old_mean_y_0) / (y_0count + 1)
                    new_var_y_0 = old_var_y_0 + ((trace - old_mean_y_0) * (trace - new_mean_y_0) - old_var_y_0) / (y_0count + 1)
                    old_mean_y_0 = new_mean_y_0
                    old_var_y_0 = new_var_y_0
                    y_0count += 1
                    count = count + 1  

                elif coeff[i][current_round*8:current_round*8+8] != '00000000':
                    new_mean_y_1 = old_mean_y_1 + (trace - old_mean_y_1) / (y_1count + 1)
                    new_var_y_1 = old_var_y_1 + ((trace - old_mean_y_1) * (trace - new_mean_y_1) - old_var_y_1) / (y_1count + 1)
                    old_mean_y_1 = new_mean_y_1
                    old_var_y_1 = new_var_y_1
                    y_1count += 1
                    count = count + 1
    # print(start_samples,end_samples)
    temp1 = old_mean_y_0 - old_mean_y_1
    temp2 = (old_var_y_0 / y_0count) + (old_var_y_1 / y_1count)
    test_result = temp1 / np.sqrt(temp2)
    # print("\n","round{} coeff{}".format(round,coeff_index))
    # print("y0",y_0count)
    # print("y1",y_1count)
    # print("rate:",y_0count/y_1count)
    return test_result


def t_test_LSB(n, url_trace,data_url,coeff_index):
    '''
    :instrctions:even  and  odd 间或采集
    :param n: blocks of traces
    :param N: sample numbers
    :return: result
    '''
    arr = np.load(url_trace.format(block_num_start))
    count = 0
    N = 300
    
    bias=0
    old_var_y_0 = np.zeros(N-bias)
    old_mean_y_0 = np.zeros(N-bias)
    old_var_y_1 = np.zeros(N-bias)
    old_mean_y_1 = np.zeros(N-bias)
    y_1count = 0
    y_0count = 0
    pos=np.load(r"D:\Dilithium_Paper_Work\Dilithium Paper\interval_data\ttest_result\Coeff_{}.npy".format(coeff_index))

    for j in trange(n):
        arr = np.load(url_trace.format(block_num_start+j))
        data = np.load(data_url.format(block_num_start+j))

        coeff=data['allbytes']

        for i in range(arr.shape[0]):
            # i是arr中曲线的数目
            for loop in range(64):
                # loop是64轮的轮数，coeff_index是第几个操作
                
                
                current_round=4*loop+coeff_index
                # 当前是256个操作中的第几个
                start_samples=pos[loop]
                end_samples=pos[loop]+N
                # 找到切片中对于的起止点

                trace=arr[i][start_samples:end_samples]

                
                if coeff[i][current_round*8:current_round*8+8][0:2] == '00':
                    # coeff长2048，每一轮长32，每一个coeff_index长8
                    # 找到第loop轮中第coeff_index为0的系数对应的曲线进行t-test
                    new_mean_y_0 = old_mean_y_0 + (trace - old_mean_y_0) / (y_0count + 1)
                    new_var_y_0 = old_var_y_0 + ((trace - old_mean_y_0) * (trace - new_mean_y_0) - old_var_y_0) / (y_0count + 1)
                    old_mean_y_0 = new_mean_y_0
                    old_var_y_0 = new_var_y_0
                    y_0count += 1
                    count = count + 1  

                elif coeff[i][current_round*8:current_round*8+8][0:2] != '00':
                    new_mean_y_1 = old_mean_y_1 + (trace - old_mean_y_1) / (y_1count + 1)
                    new_var_y_1 = old_var_y_1 + ((trace - old_mean_y_1) * (trace - new_mean_y_1) - old_var_y_1) / (y_1count + 1)
                    old_mean_y_1 = new_mean_y_1
                    old_var_y_1 = new_var_y_1
                    y_1count += 1
                    count = count + 1
    # print(start_samples,end_samples)
    temp1 = old_mean_y_0 - old_mean_y_1
    temp2 = (old_var_y_0 / y_0count) + (old_var_y_1 / y_1count)
    test_result = temp1 / np.sqrt(temp2)
    # print("\n","round{} coeff{}".format(round,coeff_index))
    # print("y0",y_0count)
    # print("y1",y_1count)
    # print("rate:",y_0count/y_1count)
    return test_result


if __name__ == '__main__':

    #block_num = 424
    block_num = 500
    block_num_start = 0

    # url_trace = r"D:\Dilithium_Paper_Work\Dilithium_paper_traces\Dilithium_paper_profile\traces_for_profiling_part{}.npy"
    url_trace = r"D:\Dilithium_Paper_Work\Dilithium_paper_traces\Dilithium_paper_attack\traces_for_attack_part{}.npy"
    # data_url=r"D:\Dilithium_Paper_Work\Dilithium_paper_metadata\metadata_files\profiling_metadata\metadata_profiling_part{}.npz"
    data_url = r"D:\Dilithium_Paper_Work\Dilithium_paper_metadata\metadata_files\attacking_metadata\metadata_attacking_part{}.npz"
    
    for i in range(4):
        result = t_test_LSB(block_num, url_trace,data_url,coeff_index=i)
        # result=np.load(r"D:\Dilithium_Paper_Work\position_and_ttest\ttest_result_coeff_{}.npy".format(i))
        top_indices = np.argsort(np.abs(result))[-24:]
        # # print(top_indices)
        plt.rcParams['figure.figsize'] = (16.0, 12.0)
        f, ax = plt.subplots(1, 1)
        line_value=4.5
        ax.axhline(y=4.5, ls='--', c='red', linewidth=2)
        ax.axhline(y=-4.5, ls='--', c='red', linewidth=2)
        
        ax.set_yticks([-120, -80, -40, -1*line_value, 0, line_value, 40, 80, 120])
        ax.set_xlabel("Samples",fontproperties = 'Times New Roman', size = 14)
        ax.set_ylabel("t-statistics",fontproperties = 'Times New Roman', size = 14)
        ax.get_yticklabels()[3].set_color("red")
        ax.get_yticklabels()[5].set_color("red")
        plt.plot(result)
        plt.show()
        # np.save(r'C:\Users\DELL\Desktop\start\ttest_result_coeff_{}.npy'.format(i), result )
    # result = np.load(r'G:/side_channel_attack/result_io_redo_share5_edit_synchronized.npy')

    