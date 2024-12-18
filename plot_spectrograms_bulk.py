# Import all dependencies
import numpy as np
from waveform_collection import gather_waveforms
from toolbox import process_waveform, plot_spectrogram_multi
from obspy import UTCDateTime, Stream
import warnings

# Define variables for functions
SOURCE = 'IRIS'
NETWORK = 'AV'
STATION = 'PN7A,PS1A,PS4A,PV6A,PVV'
LOCATION = ''
CHANNEL = '*HZ'
STARTTIME = UTCDateTime(2021, 1, 1, 0, 0)  # start time for data pull and spectrogram plot
ENDTIME = UTCDateTime(2023, 3, 1, 0, 0)  # end time for data pull and spectrogram plot
PAD = 60  # padding length [s]
CLIENT = 'IRIS'  # FDSN client for data pull
FILTER_BAND = None  # frequency band for bandpass filter
WINDOW_DURATION = 10  # spectrogram window duration [s]
FREQ_LIMS = (0.5,10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
LOG = False  # logarithmic scale in spectrogram
DEMEAN = False  # remove the temporal mean of the plotted time span from the spectrogram matrix
V_PERCENT_LIMS = (20,97.5)  # colorbar limits
EXPORT_PATH = './spectrograms/'
EXPORT_SPEC = True  # export spectrogram with no axis labels
VERBOSE = False

# Dissect time steps and loop
total_hours = int(np.floor((ENDTIME-STARTTIME)/3600))

# Commence loop
i = 0
while i < (total_hours):

    # Use try-except to catch data pull errors
    try:

        # Define time bounds for current iteration
        t1 = STARTTIME + i*3600
        t2 = STARTTIME + (i+1)*3600
        print('Now at %s...' % t1)

        # Turn off warnings if non-verbose option is selected
        if not VERBOSE:
            warnings.filterwarnings("ignore")

        # Load data using waveform_collection tool
        stream = gather_waveforms(source=SOURCE, network=NETWORK, station=STATION, location=LOCATION, channel=CHANNEL, starttime=t1-PAD, endtime=t2+PAD, verbose=False)

        # Process waveform
        stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=PAD, taper_percentage=None, filter_band=FILTER_BAND, verbose=False)
        stream_default_order = [tr.stats.station for tr in stream]
        desired_index_order = [stream_default_order.index(stn) for stn in STATION.split(',') if stn in stream_default_order]
        stream = Stream([stream[i] for i in desired_index_order])

    except:

        # Sometimes response retrieval throws out an error
        print('A data pull error occurred for i = %d, trying again...' % i)
        pass

    # Plot all spectrograms on one figure if stream is not empty
    if len(stream) != 0:
        plot_spectrogram_multi(stream, t1, t2, WINDOW_DURATION, FREQ_LIMS, log=LOG, demean=DEMEAN,
                               v_percent_lims=V_PERCENT_LIMS, earthquake_times=None, explosion_times=None,
                               db_hist=True, export_path=EXPORT_PATH, export_spec=EXPORT_SPEC)
        print('Spectrogram plotted.')
    else:
        print('Stream object is empty, skipping.')

    # Move forward in loop
    i += 1