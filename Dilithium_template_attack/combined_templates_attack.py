import numpy as np
from tqdm import *
from scipy.stats import multivariate_normal
# 假设还是读取3个模板对这仨模板的预测结果进行类似于投票的判断
N=300

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
     
    




def attack(template_url,trace_url,data_url,Z_url,position_url,blocknum):
    guess=0
    right=0
    real=0
    Gausian,poi=read_template(template_url)
    numpois=len(poi[0])
    # pos是记录起点位置的
    result=[]
    pos=[[] for _ in range(4)]
    for operation in range(4):
        op_position=np.load(position_url.format(operation))
        pos[operation]=op_position
    for block in trange(blocknum):
        trace_file=np.load(trace_url.format(block))
        data_file=np.load(data_url.format(block),allow_pickle=True)
        Z_file=np.load(Z_url.format(block))
        coeff=data_file['allbytes']
        index=data_file['index']
        Z=Z_file["Z"]
        for t_len in range(len(trace_file)):
            whole_trace=trace_file[t_len]
            for coeff_index in range(256):
                coeff_val=coeff[t_len][coeff_index*8:coeff_index*8+8]
                Z_val=Z[t_len][index[t_len]][coeff_index]
                round_num=coeff_index//4
                operation=coeff_index%4
                trace_slice=whole_trace[pos[operation][round_num]:pos[operation][round_num]+N]
                p_y=[]
                # 第coeff_index的个第operation个操作
                POI=poi[operation]
                
                # 模板coefficient的POI
                x=[trace_slice[POI[i]] for i in range(numpois)]
                for y in range(2):
                    # 将y=0，和非0的值带入多元高斯分布
                    rv=Gausian[operation][y]
                    # print(rv.pdf(x))
                    p=rv.logpdf(x)
                    # s=coeff[t_len][8*coeff_index:8*coeff_index+8]
                    p_y.append([coeff_index,y,p])
                p_y_array = np.array(p_y)  # 将 p_y 转换为 NumPy 数组
                max_p_index = np.argmax(p_y_array[:, 2])
                diff=p_y_array[0][2]-p_y_array[1][2]
                if coeff_val=="00000000":
                    real+=1
                    # print(" ",p_y_array)
                # 获取最大 p 对应的 index 和 y 值
                if max_p_index==0 and abs(Z_val)<78 and abs(diff)>10:
                    guess+=1
                    result.append(np.array([block,t_len,coeff_index]))
                    if coeff_val=="00000000":
                        right+=1

            # end for coefficient
        # ent for trace
    # end for block
    print("Real: {}, Guess: {}, Right: {}".format(real,guess,right))
    ans=np.array(result)
    np.save(r"C:\Users\DELL\Desktop\start\result_500.npy",ans)
                
                

                


        
    
        

if __name__ == '__main__':
    template_url=r"F:\Dilithium_Paper_Work\template\template_combined\75_pois\template_combined_75_coeff{}.npz"
    trace_url=r"F:\Dilithium_Paper_Work\Dilithium_paper_traces\Dilithium_paper_attack\traces_for_attack_part{}.npy"
    data_url=r"F:\Dilithium_Paper_Work\Dilithium_paper_metadata\metadata_files\attacking_metadata\metadata_attacking_part{}.npz"
    position_url=r"F:\Dilithium_Paper_Work\Dilithium Paper\core code\ttest\Coeff_{}.npy"
    Z_url=r"F:\Dilithium_Paper_Work\Dilithium_paper_metadata\Z_unpack\Attacking\Z_unpack_part{}.npz"
    attack(template_url=template_url,trace_url=trace_url,data_url=data_url,position_url=position_url,Z_url=Z_url,blocknum=500)