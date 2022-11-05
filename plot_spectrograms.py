#%% PLOT SPECTROGRAM(S)

# This script pulls in data from a local directory or queries data from a client, before processing the data and
# plotting both the waveforms and their corresponding spectrograms in separate plots.

# Import all dependencies
from toolbox import load_data, process_waveform, plot_spectrogram
from obspy import UTCDateTime
import colorcet as cc

#%% EXAMPLE 1: Plot 10 minutes of seismic and acoustic data from PS4A.
### Data are filtered in the same way (0.1 to 10 Hz) and the same window duration is used (2 s).

# Define variables for functions
network = ['AV','AV','AV','AV','AV','AV']  # SEED network code(s)
station = ['PS4A','PVV','PS1A','HAG','PN7A','PV6A']  # SEED station code(s)
channel = ['BHZ','SHZ','BHZ','SHZ','BHZ','SHZ']  # SEED channel code(s)
location = ['','','','','','']  # SEED location code(s)
starttime = UTCDateTime(2021, 10, 21, 0, 0)  # start time for data pull and spectrogram plot
endtime = starttime + 60*60  # end time for data pull and spectrogram plot
pad = 60  # padding length [s]
local = False  # pull data from local
data_dir = None  # local data directory if pulling data from local
client = 'IRIS'  # FDSN client for data pull
filter_band = (0.1,10)  # frequency band for bandpass filter
window_duration = 10  # spectrogram window duration [s]
freq_lims = (0.1,10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
log = False  # logarithmic scale in spectrogram
cmap = cc.cm.rainbow  # colormap for spectrogram
figsize = (32,9)  # figure size for output
v_percent_lims = (20,97.5)  # colorbar limits
export_path = None   # show figure in iPython

# Load data (note that network, station, channel, location can be lists of equal length)
stream = load_data(network, station, channel, location, starttime, endtime, pad=pad, local=local, data_dir=data_dir, client=client)

# Process waveform
stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=pad, taper_percentage=None, filter_band=None, verbose=True)

# Plot spectrogram for every trace in input stream
plot_spectrogram(stream, starttime, endtime, window_duration, freq_lims, log, cmap=cc.cm.rainbow,
                 v_percent_lims=v_percent_lims, figsize=figsize, export_path=export_path)

#%% EXAMPLE 2: Infrasound
### Data are filtered in the same way (0.1 to 10 Hz) as above, but a longer window duration is used (6 s).

# Define variables for functions
network = ['AV','AV','AV','AV','AV','AV']  # SEED network code(s)
station = ['PS4A','PVV','PS1A','HAG','PN7A','PV6A']  # SEED station code(s)
channel = ['BDF','BDF','BDF','BDF','BDF','BDF']  # SEED channel code(s)
location = ['','','','','','']  # SEED location code(s)
starttime = UTCDateTime(2022, 10, 22, 1, 0)  # start time for data pull and spectrogram plot
endtime = starttime + 60*60  # end time for data pull and spectrogram plot
pad = 60  # padding length [s]
local = False  # pull data from local
data_dir = None  # local data directory if pulling data from local
client = 'IRIS'  # FDSN client for data pull
filter_band = (0.1,10)  # frequency band for bandpass filter
window_duration = 10  # spectrogram window duration [s]
freq_lims = (0.1,10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
log = False  # logarithmic scale in spectrogram
cmap = cc.cm.rainbow  # colormap for spectrogram
figsize = (32,9)  # figure size for output
v_percent_lims = (20,97.5)  # colorbar limits
export_path = None   # show figure in iPython

# Load data (note that network, station, channel, location can be lists of equal length)
stream = load_data(network, station, channel, location, starttime, endtime, pad=pad, local=local, data_dir=data_dir, client=client)

# Process waveform
stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=pad, taper_percentage=None, filter_band=None, verbose=True)

# Plot spectrogram for every trace in input stream
plot_spectrogram(stream, starttime, endtime, window_duration, freq_lims, log, cmap=cc.cm.rainbow,
                 v_percent_lims=v_percent_lims, figsize=figsize, export_path=export_path)