import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import pandas as pd

from scipy import signal

counter = 0

def computeTF(start, finish, test, marker):

    impulse_start = int(4000*start)
    impulse_finish = int(4000*finish)

    hammer_data = pd.read_csv("leg_6cm" + str(test) + "hammer.csv")

    hammer_time_data = np.array(hammer_data.iloc[:,1])[impulse_start:impulse_finish] #Localised to the impulse

    output_data = pd.read_csv("leg_6cm" + str(test) + "accel.csv")

    accel_time_data = np.array(output_data.iloc[:,1])[impulse_start:impulse_finish]

    sampling_rate = 4000

    t = np.arange(0, 10, 1/sampling_rate)

    # plt.plot(t[impulse_start:impulse_finish], hammer_time_data, label='Hammer impulse ')
    # plt.plot(t[impulse_start:impulse_finish], accel_time_data, label='Accelerometer output')
    # plt.legend()
    # plt.show()

    hammer_frequency_power = fft(hammer_time_data)
    N_hammer = len(hammer_frequency_power)
    n_hammer = np.arange(N_hammer)
    T_hammer = N_hammer/sampling_rate
    freq_hammer = n_hammer/T_hammer 

    accel_frequency_power = fft(accel_time_data)
    N_accel = len(accel_frequency_power)
    n_accel = np.arange(N_accel)
    T_accel = N_accel/sampling_rate
    freq_accel = n_accel/T_accel


    transfer_function = 10*np.log10(accel_frequency_power/hammer_frequency_power)

    hammer_frequency_power = 10*np.log10(hammer_frequency_power)
    accel_frequency_power = 10*np.log10(accel_frequency_power)

    #plt.plot(freq_hammer, np.abs(hammer_frequency_power), label = 'Hammer Impulse')
    #plt.plot(freq_accel, np.abs(accel_frequency_power), label = 'Accelerometer Output')
    #plt.xlabel('Freq (Hz)')
    #plt.ylabel('FFT Amplitude |X(freq)| [dB]')
    #plt.legend()
    #plt.xlim(0, 500)
    #plt.show()
    if marker == 'x':
        counter = 1
        size = 50
    elif marker == '1':
        counter = 2
        size = 80
    elif marker == '+':
        counter = 3
        size = 80
    elif marker == '2':
        counter = 4
        size = 80
    else:
        counter = 4
        size = 80



    plt.scatter(freq_accel, np.abs(transfer_function), label = 'Test ' + str(counter), marker = marker, s=size)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('TF Amplitude')
    plt.legend()
    plt.xlim([0, 250])
    #plt.ylim([0, 5])
    #plt.show()




computeTF(4.4, 4.8, 1, 'x')
computeTF(5.4, 6.1, 1, '1')
computeTF(9.4, 9.7, 1, '+')
#computeTF(8.4, 9.0, 1, '2')
computeTF(6.85, 7.05, 1, '3')



plt.show()


# def transfer_function_welch(start, finish, test, marker):
#     impulse_start = int(4000*start)
#     impulse_finish = int(4000*finish)

    
#     sampling_rate = 4000

#     time_array = np.arange(0, finish-start, 1/sampling_rate)

#     hammer_data = pd.read_csv("leg_10cm" + str(test) + "hammer.csv")

#     hammer_time_data = np.array(hammer_data.iloc[:,1])[impulse_start:impulse_finish] #Localised to the impulse

#     output_data = pd.read_csv("leg_10cm" + str(test) + "accel.csv")

#     accel_time_data = np.array(output_data.iloc[:,1])[impulse_start:impulse_finish]


#     frequencies_csd, csds = signal.csd(hammer_time_data, accel_time_data, sampling_rate)

#     frequencies_psd, hammer_osd = signal.welch(accel_time_data, sampling_rate)


#     transfer_function = csds/hammer_osd

#     #plt.plot(frequencies_csd, transfer_function)
#     plt.plot(frequencies_csd, csds)
#     plt.xlim([0, 250])
#     plt.show()




# transfer_function_welch(1.6, 2, 1, 'x')

