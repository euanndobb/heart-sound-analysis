import pandas as pd
import numpy as np
import pydvma as dvma
from pydvma import datastructure
import datetime


dataset  = datastructure.DataSet() #This creates a blank dataset.

sampling_rate = 4000

impulse_start = int(4000*8.4)
impulse_finish = int(4000*8.9)


time_data_input = pd.read_csv("arm1hammer.csv")[impulse_start:impulse_finish] #Data array (in time domain)
time_data_output = pd.read_csv("arm1accel.csv")[impulse_start:impulse_finish] #Data array (in time domain)

time_data_input = np.array(time_data_input.iloc[:,1])
time_data_output = np.array(time_data_output.iloc[:,1])

time_samp = len(time_data_input)/sampling_rate



time_axis = np.arange(0, time_samp, 1/sampling_rate)



settings = dvma.MySettings(channels=2,
                           fs=4000,
                           stored_time=3,
                           pretrig_samples=100,
                           device_driver = 'nidaq')


t = datetime.datetime.now()
timestring = '_'+str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_at_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second)


timedata = datastructure.TimeData(time_axis, np.stack([time_data_input, time_data_output]).T , settings,timestamp=t, timestring=timestring, test_name = 'Test Input')

dataset.add_to_dataset(timedata)

dvma.save_data(dataset, 'arm1_3.npy')


