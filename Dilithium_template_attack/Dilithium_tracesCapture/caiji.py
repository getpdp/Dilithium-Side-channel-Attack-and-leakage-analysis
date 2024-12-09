from tqdm import tnrange
from tqdm.auto import tqdm
import gc
################这里是参数部分，可以修改##############################
traces_per_part = 5  # 每一块文件多少条曲线
part = 500  # 分成多少块文件
part_start_index = 0  # 序号从多少号开始。如果是初始采集，就设置为0
filename_of_traces = "F:/dilithuim采集/batch0"	# 曲线文件的文件名前缀

################这里是参数部分，可以修改##############################
global_counter = 0  # 全局的counter，用来判断这条曲线是奇数还是偶数
total_traces = part * traces_per_part  # 总共要采集的曲线数目
pbar = tqdm(total=total_traces)
def mainProcess(num_of_traces):

    traces_array = np.empty(shape=(traces_per_part,maxSamples),dtype=np.float64)
    text_in_array = np.empty(shape=(traces_per_part,33),dtype=np.uint8)
    #text_out_array = np.empty(shape=(traces_per_part,2453),dtype=np.uint8)
    text_out_array = np.empty(shape=(traces_per_part,1),dtype=np.uint8)
    

    for i in tnrange(num_of_traces):
        reset_target(scope)
        global global_counter
        global pbar
        msg_input = []
        
        status["runBlock"] = ps.ps5000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, None, 0, None, None)
        assert_pico_ok(status["runBlock"])

        ##########################################
        for m in range(33):
            # 消息
            msg_input.append(random.randint(0,255))
        #msg = [0xD8,0x1C,0x4D,0x8D,0x73,0x4F,0xCB,0xFB,0xEA,0xDE,0x3D,0x3F,0x8A,0x03,0x9F,0xAA,0x2A,0x2C,0x99,0x57,0xE8,0x35,0xAD,0x55,0xB2,0x2E,0x75,0xBF,0x57,0xBB,0x55,0x6A,0xC8]
        print(msg_input)
        target.simpleserial_write('z', bytearray(msg_input))
        
        time.sleep(1)
        #print("把plain弄成固定的，记得还原input_fixed_d_eq_1")
        #print("把key弄成全0，记得还原input_fixed_d_eq_1")
        #print("经过上面的测试，plain_mask1 - plain_mask16的结果没有问题，和keil里的一样")

        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))
        ################################
        
        bufferAMax = (ctypes.c_int16 * maxSamples)()
        bufferAMin = (ctypes.c_int16 * maxSamples)() # used for downsampling which isn't in the scope of this example
        bufferBMax = (ctypes.c_int16 * maxSamples)()
        bufferBMin = (ctypes.c_int16 * maxSamples)() # used for downsampling which isn't in the scope of this example
        source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
        status["setDataBuffersA"] = ps.ps5000aSetDataBuffers(chandle, source, ctypes.byref(bufferAMax), ctypes.byref(bufferAMin), maxSamples, 0, 0)
        assert_pico_ok(status["setDataBuffersA"])
        source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
        status["setDataBuffersB"] = ps.ps5000aSetDataBuffers(chandle, source, ctypes.byref(bufferBMax), ctypes.byref(bufferBMin), maxSamples, 0, 0)
        assert_pico_ok(status["setDataBuffersB"])
        overflow = ctypes.c_int16()
        cmaxSamples = ctypes.c_int32(maxSamples)
        
        
        status["getValues"] = ps.ps5000aGetValues(chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0, ctypes.byref(overflow))
        assert_pico_ok(status["getValues"])


        adc2mVChAMax =  adc2mV(bufferAMax, chARange, maxADC)
        adc2mVChBMax =  adc2mV(bufferBMax, chBRange, maxADC)


        # ❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗
        # 正式采集之前，先采集一条曲线的触发信号，并画图观察触发的上升沿和下降沿是否全部捕捉到，若看不到下降沿，则尝试增大postTriggerSamples
        plt.plot(adc2mVChAMax[:])	# 画图观察触发信号是否正常
        #plt.plot(adc2mVChBMax[:])	# 画图观察数据信号
        if adc2mVChBMax[0] == 32512 and adc2mVChBMax[-1] == 32512:
            raise Exception('示波器采集错误，请重新采集')
        
        recv_msg = target.simpleserial_read('*', 4)
        sm = []

#         msg = [1]
#         target.simpleserial_write('a', bytearray(msg))
#         recv_msg0 = target.simpleserial_read('1', 255)
        
#         target.simpleserial_write('b', bytearray(msg))
#         recv_msg1 = target.simpleserial_read('2', 255)
        
#         target.simpleserial_write('c', bytearray(msg))
#         recv_msg2 = target.simpleserial_read('3', 255)
        
#         target.simpleserial_write('d', bytearray(msg))
#         recv_msg3 = target.simpleserial_read('4', 255)
        
#         target.simpleserial_write('e', bytearray(msg))
#         recv_msg4 = target.simpleserial_read('5', 255)
        
#         target.simpleserial_write('f', bytearray(msg))
#         recv_msg5 = target.simpleserial_read('6', 255)
        
#         target.simpleserial_write('g', bytearray(msg))
#         recv_msg6 = target.simpleserial_read('7', 255)
        
#         target.simpleserial_write('h', bytearray(msg))
#         recv_msg7 = target.simpleserial_read('8', 255)
        
#         target.simpleserial_write('i', bytearray(msg))
#         recv_msg8 = target.simpleserial_read('9', 255)
        
#         target.simpleserial_write('j', bytearray(msg))
#         recv_msg9 = target.simpleserial_read('!', 158)

#         for x in list(recv_msg0):
#             print(hex(x))
        
#         for x in list(recv_msg1):
#             print(hex(x))
            
#         for x in list(recv_msg2):
#             print(hex(x))
            
#         for x in list(recv_msg3):
#             print(hex(x))
            
#         for x in list(recv_msg4):
#             print(hex(x))
            
#         for x in list(recv_msg5):
#             print(hex(x))
            
#         for x in list(recv_msg6):
#             print(hex(x))
            
#         for x in list(recv_msg7):
#             print(hex(x))
            
#         for x in list(recv_msg8):
#             print(hex(x))
            
#         for x in list(recv_msg9):
#             print(hex(x))
#         sm+=list(recv_msg0)
#         sm+=list(recv_msg1)
#         sm+=list(recv_msg2)
#         sm+=list(recv_msg3)
#         sm+=list(recv_msg4)
#         sm+=list(recv_msg5)
#         sm+=list(recv_msg6)
#         sm+=list(recv_msg7)
#         sm+=list(recv_msg8)
#         sm+=list(recv_msg9)
#         target.simpleserial_write('k', bytearray(msg))
#         time.sleep(0.01)
#         recv_p_0 = target.simpleserial_read('@', 255)
#         #target.flush()
        
#         target.simpleserial_write('l', bytearray(msg))
#         time.sleep(0.01)
#         recv_p_1 = target.simpleserial_read('#', 255)
#         #target.flush()
        
#         target.simpleserial_write('m', bytearray(msg))
#         time.sleep(0.01)
#         recv_p_2 = target.simpleserial_read('$', 255)
#         #target.flush()
        
#         target.simpleserial_write('n', bytearray(msg))
#         time.sleep(0.01)
#         recv_p_3 = target.simpleserial_read('%', 255)
#         #target.flush()
        
#         target.simpleserial_write('o', bytearray(msg))
#         time.sleep(0.01)
#         recv_p_4 = target.simpleserial_read('^', 255)
#         #target.flush()
#         target.simpleserial_write('p', bytearray(msg))
#         time.sleep(0.01)
#         recv_p_5 = target.simpleserial_read('&', 37)
#         #target.flush()
        
#         print(recv_p_0)
#         print(recv_p_1)
#         print(recv_p_2)
#         print(recv_p_3)
#         print(recv_p_4)
#         print(recv_p_5)

        print("结束")
        traces_array[i] = np.array(adc2mVChBMax[:])
        text_in_array[i] = np.array(msg_input)
        #text_out_array[i] = np.array(sm)
        text_out_array[i] = np.array([0])


        global_counter += 1
        #plt.xlabel('Time (ns)')
        #plt.ylabel('Voltage (mV)')
        pbar.update(1)
    return traces_array, text_in_array, text_out_array


for p in range(part):  # 遍历所有的part
    traces_arr, text_in_arr, text_out_arr = mainProcess(traces_per_part)
    np.save(filename_of_traces + "_tracesPart{0}.npy".format(p + part_start_index), traces_arr)
    np.save(filename_of_traces + "_textinPart{0}.npy".format(p + part_start_index), text_in_arr)
    np.save(filename_of_traces + "_textoutPart{0}.npy".format(p + part_start_index), text_out_arr)
    del traces_arr
    del text_in_arr
    del text_out_arr
    gc.collect()
pbar.close()