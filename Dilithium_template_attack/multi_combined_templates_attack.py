import numpy as np
from tqdm import *
from scipy.stats import multivariate_normal
# import numba as nb
N=300
GAMMA=2**32
def hamming_weight(dec_val):
    # 还原Gamma1-coefficient之前的值的HW,值不会超过17
    binary_string = bin(dec_val)[2:]
    # 计算二进制字符串中非零位的数量
    return binary_string.count('1')

def read_template(template_url):
    poi=[ []for _ in range(4)]
    Gaussian=[ [ [] for j in range(2)] for i in range(4)]
    for index in range(4):
        template=np.load(template_url.format(index),allow_pickle=True)
        mean=template['mean']
        cov=template['cov']
        poi[index].append(template['poi'])
        for y_val in range(2):
            model=multivariate_normal(mean=mean[y_val],cov=cov[y_val],allow_singular=True)
            Gaussian[index][y_val]=model
    return Gaussian,poi
     
def read_template_for_others(template_url):
    poi=[ []for _ in range(4)]
    Gaussian=[ [ [] for j in range(3)] for i in range(4)]
    for index in range(4):
        template=np.load(template_url.format(index),allow_pickle=True)
        mean=template['mean']
        cov=template['cov']
        poi[index].append(template['poi'])
        for hw in range(3):
            model=multivariate_normal(mean=mean[hw],cov=cov[hw],allow_singular=True)
            Gaussian[index][hw]=model
    return Gaussian,poi
    
def attack_core(Gausian,POI,trace_slice,coeff_index,Z_val,possibilities,gate):
    probability=[]
    numpois=len(POI)
    x=[trace_slice[POI[i]] for i in range(numpois)]
    for y in range(2):
        # 将y=0，和非0的值带入多元高斯分布
        rv=Gausian[y]
        p=rv.logpdf(x)
        probability.append([coeff_index,y,p])
    probability_arr = np.array(probability)  # 将 p_y 转换为 NumPy 数组
    max_p_index = np.argmax(probability_arr[:, 2])
    diff=probability_arr[0][2]-probability_arr[1][2]
    if max_p_index==0 and abs(diff)>gate:
        possibilities.append([coeff_index,diff,Z_val])
        return 1
    else: return 0


# 目前发现错误的都是HW在1到3之间的，建立HW为1到3的模板（这4个模板公用一套POI？）
def check_hw(Gausian_zero,Gausian_others,POI,trace_slice,coeff_index,probability,diff_gate):
    
    numpois=len(POI)
    x=[trace_slice[POI[i]] for i in range(numpois)]
        # 将y=0，和非0的值带入多元高斯分布
    rv=Gausian_zero[0]
    p=rv.logpdf(x)
    probability.append([coeff_index,0,p])
    for hw in range(3):
        rv=Gausian_others[hw]
        p=rv.logpdf(x)
        probability.append([coeff_index,hw,p])
    probability_arr = np.array(probability)  # 将 p_y 转换为 NumPy 数组
    max_p_index = np.argmax(probability_arr[:, 2])
    min=99999
    total=0
    for i in range(len(probability_arr)-1):
        diff=abs(probability_arr[0][2]-probability_arr[i+1][2])
        total+=diff
        if diff<min:
            min=diff
        # average=total/3
    # 和预测为0的预测概率差值要大，才合理
    if max_p_index==0 and min>diff_gate:
        return 1
    else: return 0

def attack(template_url1,template_url2,template_url3,template_other_url,trace_url,data_url,Z_url,position_url,start_num,blocknum,Zgate,diff_gate):
    guess=0
    right=0
    real=0
    Gausian1,poi1=read_template(template_url1)
    Gausian2,poi2=read_template(template_url2)
    Gausian3,poi3=read_template(template_url3)
    Gaussian_other,poi_other=read_template_for_others(template_other_url)
    # pos是记录起点位置的
    result=[]
    err=[]
    pos=[[] for _ in range(4)]
    for operation in range(4):
        op_position=np.load(position_url.format(operation))
        pos[operation]=op_position
    for block in trange(blocknum):
        trace_file=np.load(trace_url.format(block+start_num))
        data_file=np.load(data_url.format(block+start_num),allow_pickle=True)
        Z_file=np.load(Z_url.format(block+start_num))
        coeff=data_file['allbytes']
        index=data_file['index']
        Z=Z_file["Z"]
        for t_len in range(len(trace_file)):
            whole_trace=trace_file[t_len]
            for coeff_index in range(256):
                possibilities=[]
                coeff_val=coeff[t_len][coeff_index*8:coeff_index*8+8]
                if coeff_val=="00000000":
                        real+=1
                Z_val=Z[t_len][index[t_len]][coeff_index]
                if abs(Z_val)<Zgate:
                    round_num=coeff_index//4
                    operation=coeff_index%4
                    trace_slice=whole_trace[pos[operation][round_num]:pos[operation][round_num]+N]
                    predict1=attack_core(Gausian=Gausian1[operation],POI=poi1[operation],trace_slice=trace_slice,coeff_index=coeff_index,Z_val=Z_val,possibilities=possibilities,gate=7.55)
                    predict2=attack_core(Gausian=Gausian2[operation],POI=poi2[operation],trace_slice=trace_slice,coeff_index=coeff_index,Z_val=Z_val,possibilities=possibilities,gate=7.95)
                    predict3=attack_core(Gausian=Gausian3[operation],POI=poi3[operation],trace_slice=trace_slice,coeff_index=coeff_index,Z_val=Z_val,possibilities=possibilities,gate=8.55)
                    judge=predict1+predict2+predict3
                    
                    if judge==3:
                        probability=[]
                        check=check_hw(Gausian_zero=Gausian2[operation],Gausian_others=Gaussian_other[operation],POI=poi2[operation],
                                    trace_slice=trace_slice,coeff_index=coeff_index,probability=probability,diff_gate=diff_gate)
                        if check==1:
                            guess+=1
                            result.append(np.array([block+start_num,t_len,coeff_index]))
                            if coeff_val=="00000000":
                                right+=1
                                # print(" ",probability)
                            else:
                                # print(" ",probability)
                                wrong=[]
                                hex_string=coeff_val[6:8]+coeff_val[4:6]+coeff_val[2:4]+coeff_val[0:2]
                                if hex_string[0]=="f":
                                    dec_val=int(hex_string,16)-GAMMA
                                else:
                                    dec_val=int(hex_string,16)
                                hw=hamming_weight(abs(dec_val))
                                wrong.append([block,t_len,coeff_index])
                                wrong.append(dec_val)
                                wrong.append(hw)
                                wrong.append(hex_string)
                                if hw>3 or dec_val>0:
                                    print(wrong)
                                err.append(wrong)

            # end for coefficient
        # ent for trace
        # if block%100==0:
        #     print(" Guess Points: {} w, Real: {}, Guess: {}, Right: {}".format(block*500*256/10000,real,guess,right))
    # end for block
    print(" Guess Points: {} w, Real: {}, Guess: {}, Right: {}".format(blocknum*500*256/10000,real,guess,right))
    err_txt=r"D:\Dilithium_Paper_Work\Dilithium Paper\core code\error_file_block_Zgate_diff_gate\error_{}_z_{}_diff_{}.txt".format(blocknum,Zgate,diff_gate)
    with open(err_txt,"w") as file:
        for line in err:
            file.write(str(line)+"\n")
    
    ans=np.array(result)
    np.save(r"D:\Dilithium_Paper_Work\Dilithium Paper\core code\result_file_blocks_Z_gate_diff_gate\result_{}_z_{}_diff_{}.npy".format(blocknum,Zgate,diff_gate),ans)
                
                

                


        
    
        

if __name__ == '__main__':
    template_url1=r"D:\Dilithium_Paper_Work\template\template_combined\60_pois\template_combined_60_coeff{}.npz"
    template_url2=r"D:\Dilithium_Paper_Work\template\template_combined\75_pois\template_combined_75_coeff{}.npz"
    template_url3=r"D:\Dilithium_Paper_Work\template\template_combined\80_pois\template_combined_80_coeff{}.npz"
    template_other_url=r"D:\Dilithium_Paper_Work\template\template_for_HW_from_1_to_3\template_POI_75_coeff{}.npz"
    trace_url=r"D:\Dilithium_Paper_Work\Dilithium_paper_traces\Dilithium_paper_attack\traces_for_attack_part{}.npy"
    data_url=r"D:\Dilithium_Paper_Work\Dilithium_paper_metadata\metadata_files\attacking_metadata\metadata_attacking_part{}.npz"
    position_url=r"D:\Dilithium_Paper_Work\position_and_ttest\position\Coeff_{}.npy"
    Z_url=r"D:\Dilithium_Paper_Work\Dilithium_paper_metadata\Z_unpack\Attacking\Z_unpack_part{}.npz"
    for i in np.arange(8,14.1,0.5):
        attack(template_url1=template_url1,template_url2=template_url2,template_url3=template_url3,template_other_url=template_other_url,
           trace_url=trace_url,data_url=data_url,position_url=position_url,Z_url=Z_url,start_num=0,blocknum=9000,Zgate=27,diff_gate=i)