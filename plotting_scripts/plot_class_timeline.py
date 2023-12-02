# Import all dependencies
from obspy import UTCDateTime
from toolbox import check_timeline, plot_timeline_binned

# Define starttime and endtime
starttime = UTCDateTime(2021,12,4)
endtime = UTCDateTime(2021,12,6)

# Run check timeline to obtain class_mat
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
channel = '*HZ'
location = ''
overlap = 0.75  # 1 min overlap for 4 min interval
generate_fig = True
fig_width = 12
fig_height = 9
font_s = 8.5
model_path = './models/4min_all_augmented_revised_model.h5'
meanvar_path = './models/4min_all_augmented_revised_meanvar.npy'
class_mat, prob_mat = check_timeline(source, network, station, channel, location, starttime, endtime,
                                     model_path, meanvar_path, overlap, generate_fig=generate_fig,
                                     fig_width=fig_width, fig_height=fig_height, font_s=font_s)

# Plot binned timeline
# starttime = UTCDateTime(2021,12,4)
# endtime = UTCDateTime(2021,12,6)
classification_interval = 60     # interval used for class_mat[-1,:] or indicators [s]
binning_interval = 2 * 3600      # interval used to bin results into occurence ratios [s]
xtick_interval = 6 * 3600        # interval used to label x-axis [s]
xtick_format = '%m/%d\n%H:%M'    # compatible format for strftime
input = class_mat[-1,:]          # either class mat's last row or pickle file
class_dict = {0: ('BT', [193,  39,  45]),
              1: ('HT', [  0, 129, 118]),
              2: ('MT', [  0,   0, 167]),
              3: ('EQ', [238, 204,  22]),
              4: ('EX', [164,  98,   0]),
              5: ('NO', [ 40,  40,  40])}
cumsum_panel = True
cumsum_style = 'normalized'  # 'normalized' or 'unnormalized'
cumsum_legend = True
figsize = (10, 4.5)
fs = 12
plot_timeline_binned(starttime, endtime, classification_interval, binning_interval, xtick_interval, xtick_format,
                     input, class_dict, cumsum_panel, cumsum_style, cumsum_legend, figsize, fs)