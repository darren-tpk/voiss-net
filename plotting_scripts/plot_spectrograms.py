#%% PLOT SPECTROGRAM(S)

# Import all dependencies
from waveform_collection import gather_waveforms
from toolbox import process_waveform, rotate_NE_to_RT, plot_spectrogram, plot_spectrogram_multi
from obspy import UTCDateTime

#%% EXAMPLE 1: SEISMIC

### Define variables for functions
source = 'IRIS'
network = 'AV'
station = 'PVV'
location = ''
channel = '*H*'
starttime = UTCDateTime(2022, 10, 22, 1, 0)  # start time for data pull and spectrogram plot
endtime = starttime + 60*60  # end time for data pull and spectrogram plot
pad = 60  # padding length [s]
local = False  # pull data from local
data_dir = None  # local data directory if pulling data from local
client = 'IRIS'  # FDSN client for data pull
filter_band = None  # frequency band for bandpass filter
window_duration = 10  # spectrogram window duration [s]
freq_lims = (0.5,10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
log = False  # logarithmic scale in spectrogram
demean = False  # remove the temporal mean of the plotted time span from the spectrogram matrix
v_percent_lims = (20,97.5)  # colorbar limits
export_path = None   # show figure in iPython

# Load data using waveform_collection tool
stream = gather_waveforms(source=source, network=network, station=station, location=location, channel=channel, starttime=starttime-pad, endtime=endtime+pad)

# Process waveform
stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=pad, taper_percentage=None, filter_band=None, verbose=True)

# Convert to radial and tangential components
stream = rotate_NE_to_RT(stream,source_coord=(55.4173,-161.8937))

## Plot spectrogram for every trace in input stream
plot_spectrogram(stream, starttime, endtime, window_duration, freq_lims, log=log, demean=demean, v_percent_lims=v_percent_lims, export_path=export_path)

# Plot all spectrograms on one figure
# plot_spectrogram_multi(stream, starttime, endtime, window_duration, freq_lims, log=log, demean=demean, v_percent_lims=v_percent_lims, export_path=export_path)

#%% EXAMPLE 2: INFRASOUND

# Define variables for functions
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
location = ''
channel = '*DF'
starttime = UTCDateTime(2021, 7, 31, 15, 0)  # start time for data pull and spectrogram plot
endtime = starttime + 20*60  # end time for data pull and spectrogram plot
pad = 60  # padding length [s]
local = False  # pull data from local
data_dir = None  # local data directory if pulling data from local
client = 'IRIS'  # FDSN client for data pull
filter_band = None  # frequency band for bandpass filter
window_duration = 10  # spectrogram window duration [s]
freq_lims = (0.5,10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
log = False  # logarithmic scale in spectrogram
demean = False  # remove the temporal mean of the plotted time span from the spectrogram matrix
v_percent_lims = (20,97.5)  # colorbar limits
export_path = None   # show figure in iPython

# Load data using waveform_collection tool
stream = gather_waveforms(source=source, network=network, station=station, location=location, channel=channel, starttime=starttime-pad, endtime=endtime+pad)

# Process waveform
stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=pad, taper_percentage=None, filter_band=None, verbose=True)

## Plot spectrogram for every trace in input stream
# plot_spectrogram(stream, starttime, endtime, window_duration, freq_lims, log=log, demean=demean, v_percent_lims=v_percent_lims, export_path=export_path)

# Plot all spectrograms on one figure
plot_spectrogram_multi(stream, starttime, endtime, window_duration, freq_lims, log=log, demean=False, v_percent_lims=v_percent_lims, export_path=export_path)
