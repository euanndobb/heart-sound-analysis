
from typing import Tuple
import nidaqmx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import chirp, spectrogram


import time

class DAQInput:
    def __init__(self, name, module, channel, ch_type):
        self.name = name
        self.module = module
        self.channel = channel
        self.ch_type = ch_type
        self.signal = None
        self.fs = None
        self.bit_resolution = None


class DAQOutput:
    def __init__(self, module, channel, signal):
        self.module = module
        self.channel = channel
        self.signal = signal




def run_daq(inputs: Tuple[DAQInput], output: DAQOutput, fs: int, duration: float) -> Tuple[DAQInput]:
    system = nidaqmx.system.System.local()
    input_task = nidaqmx.Task()

    if output is not None:
        if duration is not None:
            raise ValueError("Don't specify duration when an output signal is specified")

        num_samples = len(output.signal)

        output_device = [d for d in system.devices if d.name.endswith(output.module)][0]
        output_chan = output_device.ao_physical_chans[output.channel]

        output_task = nidaqmx.Task()
        output_task.ao_channels.add_ao_voltage_chan(output_chan.name)

        output_task.timing.cfg_samp_clk_timing(
            fs,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=num_samples,
        )

        daq_fs = output_task.timing.samp_clk_rate
        if daq_fs != fs:
            raise ValueError(f"Specified sample rate {fs} not matched by output DAQ {daq_fs}")
    else:
        # No output singal, so we need to set duration of recording explicitly
        num_samples = duration * fs
        print('No output signal, recording for the specified duration')

    print('Adding input channels')
    for input in inputs:
        print('Finding device..')
        input_device = [d for d in system.devices if d.name.endswith(input.module)]
        if len(input_device) == 0:
            raise ValueError('No input DAQ module found. Is it plugged in?')
        else:
            input_device = input_device[0]
            input_chan_sensor = input_device.ai_physical_chans[input.channel]

        if input.ch_type == 'voltage':
            print('Voltage channel added') 
            input_task.ai_channels.add_ai_voltage_chan(input_chan_sensor.name)
        elif input.ch_type == 'iepe':
            print('Iepe channel added') 
            input_task.ai_channels.add_ai_force_iepe_chan(input_chan_sensor.name, sensitivity=1)
        else:
            raise ValueError("Unknown sensor type. Support 'voltage' or 'iepe'.")

    input_task.timing.cfg_samp_clk_timing(fs,
                                          sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                          samps_per_chan=num_samples)


    daq_fs = input_task.timing.samp_clk_rate
    if daq_fs != fs:
        raise ValueError(f"Specified sample rate {fs} not matched by input DAQ {daq_fs}")


    # Write output signal to DAQ box to drive phantom
    if output is not None:
        num_samples_written = output_task.write(output.signal, auto_start=True)
        if num_samples_written != len(output.signal):
            raise ValueError('Wrote less samples')

    # Start recording from DAQ box (phantom sensors, external sensors, or both)
    data = input_task.read(num_samples, timeout=(num_samples / fs) + 1)
    input_task.close()

    if output is not None:
        output_task.close()

    if len(inputs) == 1:
        # if just one input, then task.read() returns just the array instead of a list of arrays
        data = [data]

    for signal, input in zip(data, inputs):
        input.signal = signal
        input.fs = fs
        input.bit_resolution = 24

    return inputs



accel_input = DAQInput(name="Sens_1", module="2", channel="ai1", ch_type="iepe")

force_input = DAQInput(name="Sens_2", module="2", channel="ai2", ch_type="iepe")

add_accel_input = DAQInput(name="Sens_3", module="2", channel="ai3", ch_type="iepe")

#Define the output that is the shaker - I think it might use a different module here.

sampling_rate = 5120

runtime = 40 #this shoud be changed in accordance with the length of the frequency sweep.

t = np.arange(0, runtime*sampling_rate) / sampling_rate

freq_max = 500

freq_min = 10

# frequency_sweep = np.arange(10, freq_max, (freq_max - freq_min)/len(t))


arr = chirp(t, f0=freq_min, f1 = freq_max, t1 = t.max(), method = 'linear')

# zero_to_four = np.ones(sampling_rate*4)
# four_to_forty = np.linspace(1, 1.5, sampling_rate*36)


# sweep_adjustment = np.concatenate((zero_to_four, four_to_forty))

# arr = arr*sweep_adjustment



shaker_output = DAQOutput(module="3", channel = "ao0", signal = arr)

time.sleep(40)


positions = [3, 4, 5, 6, 7, 8, 9]

tests = [1, 2, 3, 4, 5] #4, 5, 6, 7, 8, 9, 10 ]#2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in positions:

    for n in tests:

        run_daq([accel_input, force_input, add_accel_input], output=shaker_output, fs=sampling_rate, duration=None)  #Change this to include a frequency sweep.


        accel_input_arr= np.array(accel_input.signal)

        force_input_arr = np.array(force_input.signal)

        add_accel_input_arr = np.array(add_accel_input.signal)


        time_arr = np.arange(0, runtime, 1/sampling_rate)


        # plt.plot(time_arr, accel_input_arr, label='Accelerometer Output')
        # plt.plot(time_arr, force_input_arr, label='Shaker Force Input')
        # #plt.plot(time, add_accel_input_arr, label='Additional Accelerometer Output')
        # plt.legend()
        # plt.show()

        name = 'chest_full_1'

        size = '19mm'

        preload = '419g'

        pd.DataFrame(force_input_arr).to_csv(name + '_' +size + '_' + preload + '_' + str(i) + '_' + str(n) + '_force.csv')
        pd.DataFrame(accel_input_arr).to_csv(name + '_' +size + '_' + preload + '_' + str(i) + '_' + str(n) +'_accel.csv')
        pd.DataFrame(add_accel_input_arr).to_csv(name + '_' +size + '_' + preload + '_' + str(i) + '_' + str(n) +'_add_accel.csv')
    
    time.sleep(15)
