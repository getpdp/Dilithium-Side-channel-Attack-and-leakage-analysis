import numpy as np
# sys.path.append(r"E:\Dilithium\dilithium采集-比赛用-最终\dilithium-py-main")
from dilithium import Dilithium,DEFAULT_PARAMETERS
import tqdm
def s1_unpack():
    D=Dilithium(DEFAULT_PARAMETERS["dilithium2"])
    s1_array=np.zeros([10,4,256],dtype=int)
    # 10个文件块，每一个s1是4*256的
    for i in range(10):
        filename=r"E:\Dilithium\dilithium采集-比赛用-最终\txts\meta_data_batch{}.txt".format(i)
        with open (filename, "r", encoding='utf-8') as f:

            for data in f.readlines():
                if data.find("sk=")!=-1:
                    sk=data[3:].strip()
                    sk_byte=bytes.fromhex(sk)
                    rho, K, tr, s1, s2, t0=D._unpack_sk_2544(sk_byte)
                    for j in range(4):
                        s1_array[i][j]=np.array(s1[j][0].coeffs,dtype=int)

    output=r"E:\Dilithium\dilithium采集-比赛用-最终\s1.npy"
    np.save(output,s1_array)

def Z_unpack(blocknum,num):
    D=Dilithium(DEFAULT_PARAMETERS["dilithium2"])
    # blocknum是块数，num是攻击集第几组
    for i in tqdm.trange(blocknum):
        filename=r"E:\Dilithium\dilithium采集-比赛用-最终\攻击集\第{}组\meta_data_part{}.npz".format(num,i)
        outpath=r"E:\Dilithium\dilithium采集-比赛用-最终\Z_unpack\第{}组\Z_unpack_part{}.npz".format(num,i)
        data=np.load(filename,allow_pickle=True)
        sm=data['sm']
        Z_array=np.zeros([len(sm),4,256],dtype=int)
        # 每一条曲线的z的多项式系数，一个z是4*256的
        C_array=np.zeros([len(sm),256],dtype=int)
        # 每一条曲线的c的多项式系数，一个z是1*256的

        for j in range(len(sm)):#对每一条曲线进行遍历
            z=bytes.fromhex(sm[j][:4840])
            C,Z_poly,h=D._unpack_sig(sig_bytes=z)
            C_tall=D._sample_in_ball(C)
            C_array[j]=np.array(C_tall.coeffs,dtype=int)
            for p in range(4):
                Z_array[j][p]=np.array(Z_poly[p][0].coeffs,dtype=int)

        np.savez_compressed(outpath,C=C_array,Z=Z_array)
        

if __name__ == '__main__':
    Z_unpack(3,8)
    # s1_unpack()
    

