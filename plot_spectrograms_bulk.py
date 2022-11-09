#%% PLOT SPECTROGRAM(S) BULK

# This script pulls in data from a local directory or queries data from a client, before processing the data and
# plotting both the waveforms and their corresponding spectrograms in separate plots.

# Import all dependencies
import numpy as np
from waveform_collection import gather_waveforms_bulk
from toolbox import process_waveform, plot_spectrogram_multi
from obspy import UTCDateTime

### Define variables for functions
volc_lat = 55.417
volc_lon = -161.894
search_radius = 25
channel = '*HZ'
starttime = UTCDateTime(2021, 8, 4, 17, 00)  # start time for data pull and spectrogram plot
endtime = UTCDateTime(2021, 8, 6, 17, 00)  # end time for data pull and spectrogram plot
pad = 60  # padding length [s]
local = False  # pull data from local
data_dir = None  # local data directory if pulling data from local
client = 'IRIS'  # FDSN client for data pull
filter_band = (0.1,10)  # frequency band for bandpass filter
window_duration = 10  # spectrogram window duration [s]
freq_lims = (0.1,10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
log = False  # logarithmic scale in spectrogram
v_percent_lims = (20,97.5)  # colorbar limits
export_path = '/Users/darrentpk/Desktop/GitHub/tremor_ml/to_label_seis/'   # show figure in iPython

# Dissect time steps and loop
total_hours = int(np.floor((endtime-starttime)/3600))

# Commence loop
for i in range(total_hours):

    # Define time bounds for current iteration
    t1 = starttime + i*3600
    t2 = starttime + (i+1)*3600

    # Load data using waveform_collection tool
    stream = gather_waveforms_bulk(lon_0=volc_lon, lat_0=volc_lat, max_radius=search_radius, starttime=t1-pad, endtime=t2+pad, channel=channel)

    # Process waveform
    stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=pad, taper_percentage=None, filter_band=None, verbose=True)

    # Plot all spectrograms on one figure
    plot_spectrogram_multi(stream,t1,t2,window_duration,freq_lims,log,v_percent_lims=v_percent_lims,export_path=export_path)