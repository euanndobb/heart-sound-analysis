import matplotlib.pyplot as plt
import numpy as np


def peak_data(data, seg_size):
    freq_res = 4000/seg_size
    low_freq = 10 #Currently both
    high_freq = 140 #Currently both
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

    return np.array(new_data1), np.array(new_data2), np.array(new_data3), np.array(new_data4)

def compute_SNR(s1_PSDs, dia_PSDs):
    SNR_averages = s1_PSDs/dia_PSDs

    return SNR_averages

def average_both_peaks(data):
    averages = []
    length = len(data)
    for n in data:
        average_var = 0
        for i in n:
            average_var += i/length
        averages.append(average_var)

    return np.array(averages)

#upload data

all_average_data = np.load('C:/Users/admin/OneDrive/4th Year Project/heart sounds analysis/heart-sounds-phantom/scripts/Euan5all_average_psds.npy')

seg_size = 390

s1_freqs, s1_PSDs, dia_freqs, dia_PSDs = peak_data(all_average_data, seg_size)  #data is a list of 16 arrays... (due to 16 points on the chest.)

SNR_averages = compute_SNR(s1_PSDs, dia_PSDs)

both_SNR_average = average_both_peaks(SNR_averages)


np.save('Euan_5_snr_averages.npy', both_SNR_average)

# for n in np.arange(0, 41, 1):

#     plt.rcParams["figure.figsize"] = (20,10)
#     plt.title('Point ' + str(n))
#     plt.semilogy(s1_freqs[n],s1_PSDs[n], color='r', label = 'S1 Average', linewidth=4)
#     plt.semilogy(dia_freqs[n], dia_PSDs[n], color='c', label = 'Diastolic Average', linewidth=4)
#     plt.xlabel('frequency [Hz]')
#     plt.ylabel('PSD [V**2/Hz]')
#     plt.xlim([0, 300])
#     plt.ylim([0.0001,10000])
#     plt.legend()
#     plt.show()

#     plt.title('Point ' + str(n))
#     plt.plot(s1_freqs[n], SNR_averages[n], label = 'SNR Average', linewidth=4)
#     plt.legend()
#     plt.show()




