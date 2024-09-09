import numpy as np
import pandas as pd
from numpy.fft import fft, ifft

from scipy import signal

import matplotlib.pyplot as plt

#Plan:

#1) Perform a frequency sweep from 0 - 250 Hz (Relvant for heart sounds)
#   Analyse both the driving-point and 'far-field' responses.
#2) take the Fourier Transforms of the input and output and compute the magnitude of the transfer function - work out what the units should be on the axes.
#   The output time data corresponding to the frequency sweep should look fairly similar to the resulting TF magnitude. 
# 3) 



def plot_tf(force_time_data, accel_time_data, add_accel_time_data, marker):

    sampling_rate = 5120
    runtime = 20

    t = np.arange(0, runtime, 1/sampling_rate)

    #Time data is now set up. It is probably worth plotting it here as check.

    # plt.title('Time Measurements')
    # plt.plot(t, force_time_data, label = 'Shaker Force')
    # plt.plot(t, accel_time_data, label = 'Accelerometer (dp)')
    # plt.plot(t, add_accel_time_data, label = 'Accelerometer (ff)')
    # plt.legend()
    # plt.show()

    force_psd= fft(force_time_data) #Work out what the psd units are...
    N_shaker = len(force_psd)
    n_shaker = np.arange(N_shaker)
    T_shaker = N_shaker/sampling_rate
    freq_shaker = n_shaker/T_shaker

    accel_psd = fft(accel_time_data)
    N_accel = len(accel_psd)
    n_accel = np.arange(N_accel)
    T_accel = N_accel/sampling_rate
    freq_accel = n_accel/T_accel

    add_accel_psd = fft(add_accel_time_data)
    N_add_accel = len(add_accel_psd)
    n_add_accel = np.arange(N_add_accel)
    T_add_accel = N_add_accel/sampling_rate
    freq_add_accel = n_add_accel/T_add_accel

    transfer_function_dp = 10*np.log10(accel_psd/force_psd)

    # hammer_frequency_power = 10*np.log10(force_psd)
    # accel_frequency_power = 10*np.log10(accel_psd)

    transfer_function_ff = 10*np.log10(add_accel_psd/force_psd)

    if marker == 'x':
        test = 'Test 1 - Small'
    elif marker == '1':
        test = 'Test 2 - Small'
    elif marker == '2':
        test = 'Test 3 - Small'
    elif marker == '3':
        test = 'Test 1 - 50g'
    elif marker == '.':
        test = 'Test 2 - 50g'
    else:
        test = 'Test 3 - 50g'



    plt.title('Transfer Functions')
    #plt.scatter(freq_accel, np.abs(transfer_function_dp), label = 'Driving Point', marker = '1', s=10)
    plt.scatter(freq_accel, np.abs(transfer_function_ff), label = test, marker = marker, s=20)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('TF Amplitude (dB)')
    plt.legend() #Relevant frequency range, check slightly above to see the affect of not sweeping to higher frequencies.
    plt.xlim([0, 200])



def plot_tf_welch(force_time_data, accel_time_data, add_accel_time_data, marker, segsize, test_number):

    sampling_rate = 5120
    runtime = 20

    t = np.arange(0, runtime, 1/sampling_rate)

    #Time data is now set up. It is probably worth plotting it here as check.

    # plt.title('Time Measurements')
    # plt.plot(t, force_time_data, label = 'Shaker Force')
    # plt.plot(t, accel_time_data, label = 'Accelerometer (dp)')
    # plt.plot(t, add_accel_time_data, label = 'Accelerometer (ff)')
    # plt.legend()
    # plt.show()

    force_psd = signal.welch(force_time_data, fs=5120, window='hann', nperseg=segsize, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

    accel_psd = signal.welch(accel_time_data, fs=5120, window='hann', nperseg=segsize, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

    add_accel_psd = signal.welch(add_accel_time_data, fs=5120, window='hann', nperseg=segsize, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    print(add_accel_psd)
    #Normalise the signals..
    
    # plt.plot(force_psd[0], force_psd[1], label = 'Force')
    # #plt.plot(accel_psd[0], accel_psd[1], label = 'Accel')
    # plt.plot(add_accel_psd[0], add_accel_psd[1], label = 'Add Accel')
    # plt.legend()
    # plt.xlim([0, 250])
    # plt.show()

    frequency_samples = force_psd[0]

    force_psd = np.array(force_psd[1])
    add_accel_psd = np.array(add_accel_psd[1])
    accel_psd = np.array(accel_psd[1])



    transfer_function_dp = 10*np.log10(accel_psd/force_psd) #The driing point repsonse.

    # hammer_frequency_power = 10*np.log10(force_psd)
    # accel_frequency_power = 10*np.log10(accel_psd)

    transfer_function_ff = 10*np.log10(add_accel_psd/force_psd) #The 'far-field' repsonse'

    plt.title('Transfer Functions')
    plt.plot(frequency_samples, np.abs(transfer_function_ff), label = 'Test ' + str(test_number)) #, marker = '1', s=50)
    #plt.plot(frequency_samples, np.abs(transfer_function_dp), label = 'Test ' + str(test_number) + ' (dp)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('TF Amplitude (dB)')
    plt.legend() #Relevant frequency range, check slightly above to see the affect of not sweeping to higher frequencies.
    plt.xlim([10, 200])
    plt.ylim([20, 65])



def plot_tf_welch_coherence(force_time_data, accel_time_data, add_accel_time_data, segsize, test_number, ax1, ax2):

    sampling_rate = 5120
    runtime = 20

    t = np.arange(0, runtime, 1/sampling_rate)

    #Time data is now set up. It is probably worth plotting it here as check.

    # plt.title('Time Measurements')
    # plt.plot(t, force_time_data, label = 'Shaker Force')
    # plt.plot(t, accel_time_data, label = 'Accelerometer (dp)')
    # plt.plot(t, add_accel_time_data, label = 'Accelerometer (ff)')
    # plt.legend()
    # plt.show()

    force_psd = signal.welch(force_time_data, fs=5120, window='hann', nperseg=segsize, noverlap=segsize//2)

    accel_psd = signal.welch(accel_time_data, fs=5120, window='hann', nperseg=segsize, noverlap=segsize//2)

    add_accel_psd = signal.welch(add_accel_time_data, fs=5120, window='hann', nperseg=segsize, noverlap=segsize//2)
    #Normalise the signals..
    
    # plt.plot(force_psd[0], force_psd[1], label = 'Force')
    # #plt.plot(accel_psd[0], accel_psd[1], label = 'Accel')
    # plt.plot(add_accel_psd[0], add_accel_psd[1], label = 'Add Accel')
    # plt.legend()
    # plt.xlim([0, 250])
    # plt.show()

    frequency_samples = force_psd[0]

    force_psd = np.array(force_psd[1])
    add_accel_psd = np.array(add_accel_psd[1])
    accel_psd = np.array(accel_psd[1])



    transfer_function_dp = 10*np.log10(accel_psd/force_psd) #The driing point repsonse.

    # hammer_frequency_power = 10*np.log10(force_psd)
    # accel_frequency_power = 10*np.log10(accel_psd)

    transfer_function_ff = 10*np.log10(add_accel_psd/force_psd) #The 'far-field' repsonse'


    input_output_coherence_ff = signal.coherence(force_time_data, add_accel_time_data, fs=sampling_rate, window='hann', nperseg=segsize, noverlap=segsize//2)
    input_output_coherence_dp = signal.coherence(force_time_data, accel_time_data, fs=sampling_rate, window='hann', nperseg=segsize, noverlap=segsize//2)


    ax1.set_xlabel('Frequency / Hz')
    ax1.set_ylabel('Transfer Function Amplitude / dB')
    #ax1.plot(frequency_samples, np.abs(transfer_function_ff), label = test_number )
    ax1.semilogx(frequency_samples, np.abs(transfer_function_dp), label = test_number + ' (dp)')
    ax1.tick_params(axis='y')
    ax1.legend(loc = 'lower right')


    ax2.set_ylabel('Coherence')  # we already handled the x-label with ax1
    ax2.semilogx(input_output_coherence_dp[0], input_output_coherence_dp[1], linestyle = 'dotted', linewidth = 2)
    ax2.tick_params(axis='y')

    return ax2


#################################################
# long exposure test

loc_1_force = "long_exposure_again_smallg_413g_1_force.csv"
loc_1_accel = "long_exposure_again_smallg_413g_1_accel.csv"
loc_1_add = "long_exposure_again_smallg_413g_1_add_accel.csv"

loc_2_force = "long_exposure_again_smallg_413g_2_force.csv"
loc_2_accel = "long_exposure_again_smallg_413g_2_accel.csv"
loc_2_add = "long_exposure_again_smallg_413g_2_add_accel.csv"

loc_3_force = "long_exposure_again_smallg_413g_3_force.csv"
loc_3_accel = "long_exposure_again_smallg_413g_3_accel.csv"
loc_3_add = "long_exposure_again_smallg_413g_3_add_accel.csv"

# loc_1_force = "chest_50g_413g_1_force.csv"
# loc_1_accel = "chest_50g_413g_1_accel.csv"
# loc_1_add = "chest_50g_413g_1_add_accel.csv"

# loc_2_force = "chest_50g_413g_2_force.csv"
# loc_2_accel = "chest_50g_413g_2_accel.csv"
# loc_2_add = "chest_50g_413g_2_add_accel.csv"

# loc_3_force = "chest_50g_413g_3_force.csv"
# loc_3_accel = "chest_50g_413g_3_accel.csv"
# loc_3_add = "chest_50g_413g_3_add_accel.csv"

# loc_4_force = "arm_smallg_413g_1_force.csv"
# loc_4_accel = "arm_smallg_413g_1_accel.csv"
# loc_4_add = "arm_smallg_413g_1_add_accel.csv"

# loc_5_force = "arm_smallg_413g_2_force.csv"
# loc_5_accel = "arm_smallg_413g_2_accel.csv"
# loc_5_add = "arm_smallg_413g_2_add_accel.csv"

# loc_6_force = "arm_smallg_413g_3_force.csv"
# loc_6_accel = "arm_smallg_413g_3_accel.csv"
# loc_6_add = "arm_smallg_413g_3_add_accel.csv"



shaker_force_data = pd.read_csv(loc_1_force)

force_time_data = np.array(shaker_force_data.iloc[:,1]) #Localised to the impulse

output_accel_data = pd.read_csv(loc_1_accel)

accel_time_data = np.array(output_accel_data.iloc[:,1])

add_accel_data = pd.read_csv(loc_1_add)

add_accel_time_data = np.array(add_accel_data.iloc[:,1])
    


shaker_force_data2 = pd.read_csv(loc_2_force)

force_time_data2 = np.array(shaker_force_data2.iloc[:,1]) #Localised to the impulse

output_accel_data2 = pd.read_csv(loc_2_accel)

accel_time_data2 = np.array(output_accel_data2.iloc[:,1])

add_accel_data2 = pd.read_csv(loc_2_add)

add_accel_time_data2 = np.array(add_accel_data2.iloc[:,1])



shaker_force_data3 = pd.read_csv(loc_3_force)

force_time_data3 = np.array(shaker_force_data3.iloc[:,1]) #Localised to the impulse

output_accel_data3 = pd.read_csv(loc_3_accel)

accel_time_data3 = np.array(output_accel_data3.iloc[:,1])

add_accel_data3 = pd.read_csv(loc_3_add)

add_accel_time_data3 = np.array(add_accel_data3.iloc[:,1])


# shaker_force_data4 = pd.read_csv(loc_4_force)

# force_time_data4 = np.array(shaker_force_data4.iloc[:,1]) #Localised to the impulse

# output_accel_data4 = pd.read_csv(loc_4_accel)

# accel_time_data4 = np.array(output_accel_data4.iloc[:,1])

# add_accel_data4 = pd.read_csv(loc_4_add)

# add_accel_time_data4 = np.array(add_accel_data4.iloc[:,1])
    


# shaker_force_data5 = pd.read_csv(loc_5_force)

# force_time_data5 = np.array(shaker_force_data5.iloc[:,1]) #Localised to the impulse

# output_accel_data5 = pd.read_csv(loc_5_accel)

# accel_time_data5 = np.array(output_accel_data5.iloc[:,1])

# add_accel_data5 = pd.read_csv(loc_5_add)

# add_accel_time_data5 = np.array(add_accel_data5.iloc[:,1])



# shaker_force_data6 = pd.read_csv(loc_6_force)

# force_time_data6 = np.array(shaker_force_data6.iloc[:,1]) #Localised to the impulse

# output_accel_data6 = pd.read_csv(loc_6_accel)

# accel_time_data6 = np.array(output_accel_data6.iloc[:,1])

# add_accel_data6 = pd.read_csv(loc_6_add)

# add_accel_time_data6 = np.array(add_accel_data6.iloc[:,1])

segsize = 5000

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# plot_tf_welch_coherence(force_time_data, accel_time_data, add_accel_time_data, segsize, 'Test 1 - 50 g', ax1, ax2)
# plot_tf_welch_coherence(force_time_data2, accel_time_data2, add_accel_time_data2, segsize, 'Test 2 - 50 g', ax1, ax2)
# plot_tf_welch_coherence(force_time_data3, accel_time_data3, add_accel_time_data3, segsize, 'Test 3 - 50 g', ax1, ax2)
# plot_tf_welch_coherence(force_time_data4, accel_time_data4, add_accel_time_data4, segsize, 'Test 1 - 25 g', ax1, ax2)
# plot_tf_welch_coherence(force_time_data5, accel_time_data5, add_accel_time_data5, segsize, 'Test 2 - 25 g', ax1, ax2)
# plot_tf_welch_coherence(force_time_data6, accel_time_data6, add_accel_time_data6, segsize, 'Test 3 - 25 g', ax1, ax2)

plot_tf_welch_coherence(force_time_data, accel_time_data, add_accel_time_data, segsize, 'Test 1', ax1, ax2)
plot_tf_welch_coherence(force_time_data2, accel_time_data2, add_accel_time_data2, segsize, 'Test 2', ax1, ax2)
plot_tf_welch_coherence(force_time_data3, accel_time_data3, add_accel_time_data3, segsize, 'Test 3', ax1, ax2)

plt.xlim([10, 200])
# ax1.set_ylim([15, 75])
ax2.set_ylim([0, 1])
plt.legend()
plt.show()






#1. Small head; 413g preload; 20 second duration.

# shaker_force_data = pd.read_csv("arm_smallg_413g_1_force.csv")

# force_time_data = np.array(shaker_force_data.iloc[:,1])#Localised to the impulse

# output_accel_data = pd.read_csv("arm_smallg_413g_1_accel.csv")

# accel_time_data = np.array(output_accel_data.iloc[:,1])

# add_accel_data = pd.read_csv("arm_smallg_413g_1_add_accel.csv")

# add_accel_time_data = np.array(add_accel_data.iloc[:,1])
    


# shaker_force_data2 = pd.read_csv("arm_smallg_413g_2_force.csv")

# force_time_data2 = np.array(shaker_force_data2.iloc[:,1])#Localised to the impulse

# output_accel_data2 = pd.read_csv("arm_smallg_413g_2_accel.csv")

# accel_time_data2 = np.array(output_accel_data2.iloc[:,1])

# add_accel_data2 = pd.read_csv("arm_smallg_413g_2_add_accel.csv")

# add_accel_time_data2 = np.array(add_accel_data2.iloc[:,1])



# shaker_force_data3 = pd.read_csv("arm_smallg_413g_3_force.csv")

# force_time_data3 = np.array(shaker_force_data3.iloc[:,1])#Localised to the impulse

# output_accel_data3 = pd.read_csv("arm_smallg_413g_3_accel.csv")

# accel_time_data3 = np.array(output_accel_data3.iloc[:,1])

# add_accel_data3 = pd.read_csv("arm_smallg_413g_3_add_accel.csv")

# add_accel_time_data3 = np.array(add_accel_data3.iloc[:,1])

# segsize = 10000

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# plot_tf_welch_coherence(force_time_data, accel_time_data, add_accel_time_data, segsize, 1, ax1, ax2)
# plot_tf_welch_coherence(force_time_data2, accel_time_data2, add_accel_time_data2, segsize, 2, ax1, ax2)
# plot_tf_welch_coherence(force_time_data3, accel_time_data3, add_accel_time_data3, segsize, 3, ax1, ax2)

# plt.xlim([10, 200])
# ax1.set_ylim([20, 65])
# ax2.set_ylim([0, 1])
# plt.legend()
# plt.show()
