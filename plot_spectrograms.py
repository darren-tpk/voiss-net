#%% PLOT SPECTROGRAM(S)

# This script pulls in data from a local directory or queries data from a client, before processing the data and
# plotting both the waveforms and their corresponding spectrograms in separate plots.

# Import all dependencies
from toolbox import load_data, process_waveform, plot_spectrogram
from obspy import UTCDateTime

# Define variables for functions
network = ['AV','AV']  # SEED network code(s)
station = ['PVV','PVV']  # SEED station code(s)
channel = ['SHZ','BDF']  # SEED channel code(s)
location = ['','']  # SEED location code(s)
starttime = UTCDateTime(2022, 5, 23, 8, 30)  # start time for data pull and spectrogram plot
endtime = UTCDateTime(2022, 5, 23, 8, 40)  # end time for data pull and spectrogram plot
pad = 60  # padding length [s]
local = False  # pull data from local
data_dir = None  # local data directory if pulling data from local
client = 'IRIS'  # FDSN client for data pull
filter_band = (1,10)  # frequency band for bandpass filter
window_duration = 2  # spectrogram window duration [s]
freq_lims = None  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
log = False  # logarithmic scale in spectrogram
export_path = None  # show figure in iPython

# Load data (note that network, station, channel, location can be lists of equal length)
stream = load_data(network, station, channel, location, starttime, endtime, pad=pad, local=local, data_dir=data_dir, client=client)

# Process waveform
stream = process_waveform(stream, remove_response=True, detrend=True, taper_length=pad, taper_percentage=None, filter_band=(1, 10), verbose=True)

# Plot spectrogram for every trace in input stream
plot_spectrogram(stream,starttime,endtime,window_duration,freq_lims,log,export_path=None)