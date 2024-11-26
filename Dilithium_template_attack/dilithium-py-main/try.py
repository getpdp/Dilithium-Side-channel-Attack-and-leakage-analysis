import sys
sys.path.append(r"C:\Users\DELL\Desktop\dilithium-py-main\dilithium-py-main")
from modules import *
from dilithium import Dilithium,DEFAULT_PARAMETERS
import numpy as np
from polynomials import*
f=np.load("metadata_profiling_part0.npz",allow_pickle=True)
sm_msg=f['sm']
sk=f['sk']
sk_sample=sk[0]
z=bytes.fromhex(sm_msg[0][:4840])
sk_byte=bytes.fromhex(sk_sample)
# print(len(sk_byte))
D=Dilithium(DEFAULT_PARAMETERS["dilithium2"])
C,Z_poly,h=D._unpack_sig(sig_bytes=z)
rho, K, tr, s1, s2, t0=D._unpack_sk_2544(sk_byte)
# print(Z_poly)
# print((s1[0][0].coeffs))
C_tall=D._sample_in_ball(C)
f1=np.load("E:\Dilithium\dilithium采集-比赛用-最终\Z_unpack\第9组\Z_unpack_part0.npz")
print(f1['C'].shape)
# print(C_tall.coeffs)
# f2=np.load("E:\Dilithium\dilithium采集-比赛用-最终\s1.npy")
# print(f2.shape)