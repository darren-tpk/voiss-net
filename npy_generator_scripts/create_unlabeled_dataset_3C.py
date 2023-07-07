# %% CREATE DATASET

# Import all dependencies
import numpy as np
from obspy import UTCDateTime
from waveform_collection import gather_waveforms
from toolbox import process_waveform, calculate_spectrogram, rotate_NE_to_RT

# Define filepaths and variables for functions
starttime = UTCDateTime(2022, 9, 23, 00, 00)  # start time for data pull and spectrogram plot
endtime = UTCDateTime(2023, 1, 1, 00, 00)  # end time for data pull and spectrogram plot
time_step = 4 * 60  # Create a training dataset with 2D matrices spanning 4 minutes each
output_dir = '/Users/darrentpk/Desktop/all_npy_3C/'
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
location = ''
channel = '*H*'
pad = 240  # padding length [s]
local = False  # pull data from local
data_dir = None  # local data directory if pulling data from local
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

    # Loop over stream stations
    station_list = [tr.stats.station for tr in stream]
    stream_stations = list(set(station_list))
    for j, stream_station in enumerate(stream_stations):

        # Choose traces corresponding to station
        station_traces = [tr for tr in stream if tr.stats.station == stream_station]

        # Implement check to see if there is both horizontal and vertical
        components_seen = [tr.stats.channel[-1] for tr in station_traces]
        if set(components_seen) != set(['N','E','Z']):
            print('Station %s does not have 3 component traces available at this time period. Skipping.' % stream_station)
            continue

        # Rotate and re-extract components
        station_coord = (55.4173, -161.8937)  # pavlof volcano
        station_traces_rotated = rotate_NE_to_RT(station_traces, station_coord)
        components_seen = [tr.stats.channel[-1] for tr in station_traces_rotated]

        # Calculate spectrogram power matrix and stack
        spec_db_Z, utc_times = calculate_spectrogram(station_traces_rotated[components_seen.index('Z')], t1, t2, window_duration, freq_lims, demean=demean)
        spec_db_T, _ = calculate_spectrogram(station_traces_rotated[components_seen.index('T')], t1, t2, window_duration, freq_lims, demean=demean)
        spec_db_R, _ = calculate_spectrogram(station_traces_rotated[components_seen.index('R')], t1, t2, window_duration, freq_lims, demean=demean)

        # Define array of time steps for spectrogram slicing
        step_bounds = np.arange(t1, t2+time_step, time_step)

        # Loop over time steps
        for k in range(len(step_bounds) - 1):

            # Slice spectrogram
            sb1 = step_bounds[k]
            sb2 = step_bounds[k + 1]
            spec_slice_indices = np.flatnonzero([sb1 < t < sb2 for t in utc_times])
            spec_slice_Z = spec_db_Z[:, spec_slice_indices]
            spec_slice_T = spec_db_T[:, spec_slice_indices]
            spec_slice_R = spec_db_R[:, spec_slice_indices]

            # Enforce shape
            if np.shape(spec_slice_Z) != (94, time_step):
                # Try inclusive slicing time span (<= sb2)
                spec_slice_indices = np.flatnonzero([sb1 < t <= sb2 for t in utc_times])
                spec_slice_Z = spec_db_Z[:, spec_slice_indices]
            # Enforce shape
            if np.shape(spec_slice_T) != (94, time_step):
                # Try inclusive slicing time span (<= sb2)
                spec_slice_indices = np.flatnonzero([sb1 < t <= sb2 for t in utc_times])
                spec_slice_T = spec_db_T[:, spec_slice_indices]
            # Enforce shape
            if np.shape(spec_slice_R) != (94, time_step):
                # Try inclusive slicing time span (<= sb2)
                spec_slice_indices = np.flatnonzero([sb1 < t <= sb2 for t in utc_times])
                spec_slice_R = spec_db_R[:, spec_slice_indices]

            # Stack spec slices
            spec_slice_stack = np.dstack([spec_slice_Z, spec_slice_R, spec_slice_T])

            # Skip matrices that have a spectrogram data gap
            if np.sum(spec_slice_stack.flatten() < -220) > 150:
                print('Skipping due to data gap, %d elements failed the check' % np.sum(spec_slice_stack.flatten() < -220))
                continue

            # Craft filename and save
            file_name = stream_station + '_' + sb1.strftime('%Y%m%d%H%M') + '_' + \
                        sb2.strftime('%Y%m%d%H%M') + '_3C.npy'
            if file_name not in current_files:
                np.save(output_dir + file_name, spec_slice_stack)
