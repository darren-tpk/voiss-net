# %% CREATE DATASET

# Import all dependencies
import json
import numpy as np
import statistics as st
from obspy import UTCDateTime
from waveform_collection import gather_waveforms
from toolbox import process_waveform, calculate_spectrogram

# Define filepaths and variables for functions
starttime = UTCDateTime(2023, 1, 1, 0, 0, 0)  # start time for data pull and spectrogram plot
endtime = UTCDateTime(2023, 3, 1, 0, 0, 0)  # end time for data pull and spectrogram plot
time_step = 4 * 60  # Create a training dataset with 2D matrices spanning 4 minutes each
output_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2021_2022_npy/'
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
location = ''
channel = '*HZ'
pad = 60  # padding length [s]
client = 'IRIS'  # FDSN client for data pull
filter_band = None  # frequency band for bandpass filter
window_duration = 10  # spectrogram window duration [s]
freq_lims = (0.5, 10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
demean = False  # remove the temporal mean of the plotted time span from the spectrogram matrix

# Calculate number of days for loop
num_days = int((endtime-starttime)/86400)

# Find current files
import glob
current_files = glob.glob(output_dir + '*.npy')
current_files = [c.split('/')[-1] for c in current_files]

# Loop over days
for i in range(num_days):

    t1 = starttime + (i*86400)
    t2 = t1 + 86400
    print('Now at %s ...' % t1)

    # Load data using waveform_collection tool
    successfully_loaded = False
    while not successfully_loaded:
        try:
            # Gather waveforms
            stream = gather_waveforms(source=source, network=network, station=station, location=location,
                                      channel=channel, starttime=t1 - pad, endtime=t2 + pad, verbose=False)

            # Process waveform
            stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=pad,
                                      taper_percentage=None, filter_band=filter_band, verbose=False)
            successfully_loaded = True
        except:
            print('gather_waveforms failed to retrieve response')
            pass

    # Loop over stations that have data
    stream_stations = [tr.stats.station for tr in stream]
    for j, stream_station in enumerate(stream_stations):

        # Choose trace corresponding to station
        trace = stream[j]

        # Calculate spectrogram power matrix
        spec_db, utc_times = calculate_spectrogram(trace, t1, t2, window_duration, freq_lims, demean=demean)

        # Define array of time steps for spectrogram slicing
        step_bounds = np.arange(t1, t2+time_step, time_step)

        # Loop over time steps
        for k in range(len(step_bounds) - 1):

            # Slice spectrogram
            sb1 = step_bounds[k]
            sb2 = step_bounds[k + 1]
            spec_slice_indices = np.flatnonzero([sb1 < t < sb2 for t in utc_times])
            spec_slice = spec_db[:, spec_slice_indices]

            # Enforce shape
            if np.shape(spec_slice) != (94, time_step):
                # Try inclusive slicing time span (<= sb2)
                spec_slice_indices = np.flatnonzero([sb1 < t <= sb2 for t in utc_times])
                spec_slice = spec_db[:, spec_slice_indices]
                # If it still doesn't fit our shape, raise error
                if np.shape(spec_slice) != (94, time_step):
                    raise ValueError('THE SHAPE IS NOT RIGHT.')

            # Skip matrices that have a spectrogram data gap
            if np.sum(spec_slice.flatten() < -220) > 50:
                print('Skipping due to data gap, %d elements failed the check' % np.sum(spec_slice.flatten() < -220))
                continue

            # Craft filename and save
            file_name = stream_station + '_' + sb1.strftime('%Y%m%d%H%M') + '_' + \
                        sb2.strftime('%Y%m%d%H%M') + '.npy'
            if file_name not in current_files:
                np.save(output_dir + file_name, spec_slice)