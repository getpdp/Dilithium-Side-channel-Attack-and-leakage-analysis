import numpy as np
import matplotlib.pyplot as plt
def SNR_data_Mu(trace_url,x_url,blocknum,types,samples):
    mu_data=np.zeros([types,samples])
    # data types:256, trace samples:500
    
    counter=np.zeros(types)
    for block in range(blocknum):
        x=np.load(x_url.format(block))&0xff
        trace=np.load(trace_url.format(block))
        for t in range(trace.shape[0]):
            counter[x[t]]+=1
            mu_data[x[t]]=((counter[x[t]]-1)*mu_data[x[t]]+trace[t])/counter[x[t]]
    return mu_data

def SNR_noise_Var(trace_url,x_url,blocknum,types,samples,mu_data):
    mu_noise=np.zeros(samples)
    mu_noise_square=np.zeros(samples)
    counter=0
    for block in range(blocknum):
        x=np.load(x_url.format(block))&0xff
        trace=np.load(trace_url.format(block))
        for t in range(trace.shape[0]):
            counter+=1
            mu_noise=((counter-1)*mu_noise+trace[t]-mu_data[x[t]])/counter
            mu_noise_square=((counter-1)*mu_noise_square+(trace[t]-mu_data[x[t]])**2)/counter
    var_noise=mu_noise_square-mu_noise
    return var_noise

def SNR(trace_url,x_url,blocknum,types,samples):
    mu_data=SNR_data_Mu(trace_url,x_url,blocknum,types,samples)
    var_noise=SNR_noise_Var(trace_url,x_url,blocknum,types,samples,mu_data)
    var_data=np.var(mu_data,axis=0)
    SNR_arr=var_data/var_noise
    np.save(r"",SNR_arr)
    plt.plot(SNR_arr)
    plt.show()
    print(SNR_arr)


if __name__ =="__main__":
    trace=r""
    x=r""
    SNR(trace,x,20,types=256,samples=500)
