# Import all dependencies
from toolbox import load_data, process_waveform, plot_spectrogram
from obspy import UTCDateTime
import glob
from obspy import UTCDateTime, read, Stream, Trace
import numpy as np
from matplotlib import dates
from scipy.signal import spectrogram

# Define variables for functions
network = ['AV','AV','AV','AV','AV','AT']  # SEED network code(s)
station = ['PV6A','PS4A','PVV','HAG','PS1A','SDPT']  # SEED station code(s)
channel = ['SHZ','BHZ','SHZ','SHZ','BHZ','BHZ']  # SEED channel code(s)
location = ['','','','','','']  # SEED location code(s)
starttime = UTCDateTime(2022, 7, 22, 14, 10)  # start time for data pull and spectrogram plot
endtime = starttime + 10*60  # end time for data pull and spectrogram plot
pad = 60  # padding length [s]
local = False  # pull data from local
data_dir = None  # local data directory if pulling data from local
client = 'IRIS'  # FDSN client for data pull
filter_band = (0.1,10)  # frequency band for bandpass filter
window_duration = 2  # spectrogram window duration [s]
freq_lims = (0.1,10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
log = False  # logarithmic scale in spectrogram
export_path = None  # show figure in iPython

# Load data (note that network, station, channel, location can be lists of equal length)
stream = load_data(network, station, channel, location, starttime, endtime, pad=pad, local=local, data_dir=data_dir, client=client)

# Process waveform
stream = process_waveform(stream, remove_response=True, detrend=True, taper_length=pad, taper_percentage=None, filter_band=filter_band, verbose=True)

# Plot spectrogram for every trace in input stream
plot_spectrogram(stream, starttime, endtime, window_duration, freq_lims, log, cmap='jet', export_path='/Users/darrentpk/Desktop/GitHub/tremor_ml/sample_spectrograms/')

from matplotlib import pyplot as plt
import pandas as pd
from glob import glob
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('Qt5Agg')
directory = '/Users/darrentpk/Desktop/GitHub/tremor_ml/sample_spectrograms/'
all_images = glob(directory + '*Z.png')  # glob to get list of all the images to classify
results = pd.DataFrame(columns=["image_file", "Classification"])  # can add other columns of metadata
# results = pd.read_csv('clev_201718_classifications.csv')
# print("Starting at {}".format(granules[0]))
for img_file in all_images:
    try:
        img = mpimg.imread(img_file)  # open images with gdal, convert to nparray
    except IOError:
        print("unable to open {}".format(img_file))
        continue

    # now create the initial plot:
    plt.imshow(img)
    plt.colorbar()  # add a colorscale
    plt.title(img_file)  # add the filename
    coords = plt.ginput(n=-1, timeout=-1)  # this line waits for your input, see documentation probably

    # make sense of your input:
    row = {"image_file": img_file}  # dictionary to hold image info, before updating to results dataframe
    if len(coords) == 0:  # if no clicks on the image
        row["Classification"] = "Active"
    elif len(coords) >= 3:  # if >=3 clicks
        row["Classification"] = 'Ambiguous'
    else:  # if 1 click (or 2 on accident to be safe)
        row["Classification"] = "Inactive"

    plt.close()  # must close plot because loop will automatically open the next image
    results = results.append(row, ignore_index=True)  # update results dataframe

results.to_csv('/Users/darrentpk/Desktop/GitHub/tremor_ml/classification.csv', index=False)  # save once more, out of the loop