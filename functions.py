from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def intervals_clean2(starts, ends, seg_size):
    ends =np.delete(ends, [0, len(ends)-1])
    starts =np.delete(starts, [0, len(starts)-1])
    length = len(ends)
    delete_list = []
    delete_list2 = []
    for i in range(length):
        if ends[i] - starts[i] < seg_size:
            delete_list.append(ends[i])
            delete_list2.append(starts[i])

    ends = ends.tolist()
    starts = starts.tolist()
    for a in delete_list:
        ends.remove(a)
    for b in delete_list2:
        starts.remove(b)

    ends = np.array(ends)
    starts = np.array(starts)

    
    return starts, ends

def averagePSD(all_f, all_PSD):
    averages = []
    for n in range(all_f.shape[1]):
        average_var = 0
        for i in range(all_f.shape[0]):
            average_var += all_PSD[i][n]/all_PSD.shape[0]
        averages.append(average_var)

    return all_f[0], averages

def welchsPSD(data, start, end, all_f, all_PSD, seg_size):
    sampling_freq = 4000
    data_interest = data[start:end]

    f, PSDs = signal.welch(data_interest, sampling_freq, nperseg=seg_size)
    all_f.append(f)
    all_PSD.append(PSDs)

    #plt.semilogy(f, PSDs)

    return f, PSDs, all_f, all_PSD



def peak_data(data, seg_size):
    freq_res = 4000/seg_size
    low_freq = 0
    high_freq = 100 #These are subject to change...
    low_freq = int(np.floor(low_freq/freq_res))
    high_freq = int(np.ceil(high_freq/freq_res) + 1)
    new_data1 = []
    new_data2 = []
    new_data3 = []
    new_data4 = []
    for n in data:
        new_data1.append(n[0][low_freq:high_freq])
        new_data2.append(n[1][low_freq:high_freq])
        new_data3.append(n[2][low_freq:high_freq])
        new_data4.append(n[3][low_freq:high_freq])

    return new_data1, new_data2, new_data3, new_data4







