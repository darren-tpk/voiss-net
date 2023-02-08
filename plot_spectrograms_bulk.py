#%% PLOT SPECTROGRAM(S) BULK

# Import all dependencies
import numpy as np
from waveform_collection import gather_waveforms
from toolbox import process_waveform, plot_spectrogram, plot_spectrogram_multi
from obspy import UTCDateTime
from libcomcat.search import search
import warnings

### Define variables for functions
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
location = ''
channel = '*HZ'
starttime = UTCDateTime(2021, 7, 1, 00, 00)  # start time for data pull and spectrogram plot
endtime = UTCDateTime(2021, 7, 22, 00, 00)  # end time for data pull and spectrogram plot
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
export_path = '/Users/darrentpk/Desktop/spectrograms_20210701_20210722/' # '/Users/darrentpk/Desktop/to_label_seis/'   # show figure in iPython
export_spec = True  # export spectrogram with no axis labels
verbose = False

# Dissect time steps and loop
total_hours = int(np.floor((endtime-starttime)/3600))

# Commence loop
i = 0
while i < (total_hours-1):

    try:

        # Define time bounds for current iteration
        t1 = starttime + i*3600
        t2 = starttime + (i+1)*3600

        # search for earthquakes
        volc_lat = 55.417
        volc_lon = -161.894
        maxradiuskm = 275
        earthquakes = search(starttime=t1.datetime,
                             endtime=t2.datetime,
                             latitude=volc_lat,
                             longitude=volc_lon,
                             maxradiuskm=maxradiuskm,
                             reviewstatus='reviewed')
        eq_times = [UTCDateTime(eq.time) for eq in earthquakes]

        # Turn off warnings if non-verbose option is selected
        if not verbose:
            warnings.filterwarnings("ignore")

        # Load data using waveform_collection tool
        stream = gather_waveforms(source=source, network=network, station=station, location=location, channel=channel, starttime=t1-pad, endtime=t2+pad, verbose=False)

        # Process waveform
        stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=pad, taper_percentage=None, filter_band=filter_band, verbose=False)

        # Plot all spectrograms on one figure
        plot_spectrogram_multi(stream, t1, t2, window_duration, freq_lims, log=log, demean=demean, v_percent_lims=v_percent_lims, earthquake_times=eq_times, db_hist=True, export_path=export_path, export_spec=export_spec)

        i += 1

    except:

        # Sometimes response retrieval throws out an error (?)
        print('some error happened for i = %d, trying again' % i)
        pass