# import time
# import pickle
# import os
# import glob
# import colorcet as cc
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# from obspy import UTCDateTime, read, Stream, Trace
# from matplotlib import dates
# from matplotlib.transforms import Bbox
# from scipy.signal import spectrogram, find_peaks
# from scipy.fft import rfft, rfftfreq
# from ordpy import complexity_entropy
# from geopy.distance import geodesic as GD
# from matplotlib.colors import ListedColormap
# from DataGenerator import DataGenerator
# from keras.models import load_model
# from waveform_collection import gather_waveforms
# from toolbox import process_waveform, calculate_spectrogram
#
# starttime = starttime
# endtime = endtime
# time_step = 60
#
# def plot_timeline(starttime,endtime,time_step,type,model_path,indicators_path,plot_title,export_path=None,transparent=False,plot_labels=False,labels_kwargs=None):
#     """
#     Plot timeline figure showing station-specific and probability-sum voting by monthly rows
#     :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`):
#     :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`):
#     :param time_step (float): time step used to divide plot into columns
#     :param type (str): defined as either 'seismic' or 'infrasound' to determine color palette
#     :param model_path (str): path to model .h5 file
#     :param indicators_path (str): path to pkl file that stores timeline indicators for plotting
#     :param plot_title (str): plot title to use
#     :param export_path (str): filepath to export figures (condensed plot will tag "_condensed" to filename)
#     :param transparent (bool): if `True`, export figures with transparent background
#     :param plot_labels (bool): if `True`, plot manual timeframe with manual labels. Requires `labels_kwargs`.
#     :param labels_kwargs (dict): dictionary with the keys `start_date` (UTCDateTime), `end_date` (UTCDateTime) and `labels_dir` (str)
#     :return: None
#     """
#
#     print('Plotting timeline...')
#
#     # Load indicators
#     with open(indicators_path, 'rb') as f:  # Unpickling
#         indicators = pickle.load(f)
#
#     # Determine number of stations and number of months
#     stations = list(np.unique([i[0] for i in indicators]))
#     nsubrows = len(stations)
#     month_list = []
#     month_utcdate = starttime
#     while month_utcdate < endtime:
#         month_list.append(month_utcdate.strftime('%b \'%y'))
#         month_utcdate += 31 * 86400
#         month_utcdate = UTCDateTime(month_utcdate.year, month_utcdate.month, 1)
#     nmonths = len(month_list)
#
#     # Load model to determine number of classes
#     saved_model = load_model(model_path)
#     nclasses = saved_model.layers[-1].get_config()['units']
#     na_label = nclasses
#
#     # Define fixed params
#     TICK_DAYS = [0, 7, 14, 28]
#
#     # Craft unlabeled matrix and store probabilities
#     matrix_length = int(31 * (86400 / time_step))
#     matrix_height = nmonths * nsubrows
#     matrix_plot = np.ones((matrix_height, matrix_length)) * na_label
#     matrix_probs = np.zeros((matrix_height, matrix_length, nclasses))
#     for indicator in indicators:
#         utc = indicator[1]
#         row_index = nsubrows * month_list.index(utc.strftime('%b \'%y')) + stations.index(indicator[0])
#         col_index = int((indicator[1] - UTCDateTime(utc.year, utc.month, 1)) / time_step)
#         matrix_plot[row_index, col_index] = indicator[2]
#         matrix_probs[row_index, col_index, :] = indicator[3]
#
#     # Craft labeled matrix
#     if plot_labels:
#         labeled_start_index = int((labels_kwargs['start_date'] - UTCDateTime(labels_kwargs['start_date'].year,
#                                                                              labels_kwargs['start_date'].month,
#                                                                              1)) / time_step)
#         labeled_end_index = int((labels_kwargs['end_date'] - UTCDateTime(labels_kwargs['end_date'].year,
#                                                                          labels_kwargs['end_date'].month,
#                                                                          1)) / time_step)
#         labeled_start_row = int(np.floor((labels_kwargs['start_date'] - starttime) / (31 * 86400)))
#         labeled_end_row = int(np.floor((labels_kwargs['end_date'] - starttime) / (31 * 86400)))
#         labeled_matrix_plot = np.ones((matrix_height, matrix_length)) * na_label
#         labeled_spec_paths = glob.glob(labels_kwargs['labels_dir'] + '*.npy')
#         labeled_indicators = []
#         for i, filepath in enumerate(labeled_spec_paths):
#             filename = filepath.split('/')[-1]
#             chunks = filename.split('_')
#             labeled_indicators.append([chunks[0], UTCDateTime(chunks[1]), int(chunks[3][0])])
#         for labeled_indicator in labeled_indicators:
#             utc = labeled_indicator[1]
#             row_index = nsubrows * month_list.index(utc.strftime('%b \'%y')) + stations.index(labeled_indicator[0])
#             col_index = int((labeled_indicator[1] - UTCDateTime(utc.year, utc.month, 1)) / time_step)
#             labeled_matrix_plot[row_index, col_index:col_index + int(240 / time_step)] = labeled_indicator[2]
#
#     # Choose color palette depending on data type and nclasses
#     if type == 'seismic':
#         # Craft corresponding rgb values
#         if nclasses == 6:
#             rgb_values = np.array([
#                 [193, 39, 45],
#                 [0, 129, 118],
#                 [0, 0, 167],
#                 [238, 204, 22],
#                 [164, 98, 0],
#                 [40, 40, 40],
#                 [255, 255, 255]])
#             rgb_keys = ['Broadband\nTremor',
#                         'Harmonic\nTremor',
#                         'Monochromatic\nTremor',
#                         'Non-Tremor\nSignal',
#                         'Explosion',
#                         'Noise',
#                         'N/A']
#         elif nclasses == 7:
#             rgb_values = np.array([
#                 [193, 39, 45],
#                 [0, 129, 118],
#                 [0, 0, 167],
#                 [238, 204, 22],
#                 [103, 72, 132],
#                 [164, 98, 0],
#                 [40, 40, 40],
#                 [255, 255, 255]])
#             rgb_keys = ['Broadband\nTremor',
#                         'Harmonic\nTremor',
#                         'Monochromatic\nTremor',
#                         'Non-Tremor\nSignal',
#                         'Long\nPeriod',
#                         'Explosion',
#                         'Noise',
#                         'N/A']
#     else:
#         # Craft corresponding rgb values
#         rgb_values = np.array([
#             [103, 52, 235],
#             [235, 152, 52],
#             [40, 40, 40],
#             [15, 37, 60],
#             [255, 255, 255]])
#         rgb_keys = ['Infrasonic\nTremor',
#                     'Explosion',
#                     'Wind\nNoise',
#                     'Electronic\nNoise',
#                     'N/A']
#     rgb_ratios = rgb_values / 255
#     colors = np.concatenate((rgb_ratios, np.ones((np.shape(rgb_values)[0], 1))), axis=1)
#     cmap = ListedColormap(colors)
#     if plot_labels:
#         colors[-1][-1] = 0
#         labeled_cmap = ListedColormap(colors)
#
#     # Colorbar keywords
#     real_cbar_tick_interval = 2 * (len(np.unique(matrix_plot)) - 1) / (2 * np.shape(rgb_values)[0])
#     real_cbar_ticks = np.arange(real_cbar_tick_interval / 2, len(np.unique(matrix_plot)) - 1, real_cbar_tick_interval)
#     cbar_kws = {'ticks': real_cbar_ticks,
#                 'drawedges': True,
#                 'aspect': 30}
#
#     # Craft timeline figure
#     fig, ax = plt.subplots(figsize=(20, nmonths * nsubrows / 3))
#     sns.heatmap(matrix_plot, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.8, vmin=0, vmax=nclasses)
#     if plot_labels:
#         sns.heatmap(labeled_matrix_plot, cmap=labeled_cmap, cbar=False)
#     cbar = ax.collections[0].colorbar
#     cbar.outline.set_color('black')
#     cbar.outline.set_linewidth(1.5)
#     cbar.ax.set_yticklabels(rgb_keys, fontsize=22)
#     cbar.ax.invert_yaxis()
#     for y in range(0, matrix_height, nsubrows):
#         ax.axhline(y=y, color='black')
#         for i, station in enumerate(stations):
#             ax.text(100, y + i + 1, station, fontsize=15)
#     if plot_labels:
#         ax.plot([labeled_start_index, matrix_length], [labeled_start_row * nsubrows, labeled_start_row * nsubrows],
#                 'k-', linewidth=5.5)
#         ax.plot([labeled_start_index, matrix_length],
#                 [(labeled_start_row + 1) * nsubrows, (labeled_start_row + 1) * nsubrows], 'k-', linewidth=5.5)
#         for r in range(labeled_start_row, labeled_end_row):
#             ax.plot([0, matrix_length], [(r + 1) * nsubrows, (r + 1) * nsubrows], 'k-', linewidth=5.5)
#         ax.plot([0, labeled_end_index], [labeled_end_row * nsubrows, labeled_end_row * nsubrows], 'k-', linewidth=5.5)
#         ax.plot([0, labeled_end_index], [(labeled_end_row + 1) * nsubrows, (labeled_end_row + 1) * nsubrows], 'k-',
#                 linewidth=5.5)
#     ax.set_yticks(np.arange(nsubrows / 2, matrix_height, nsubrows))
#     ax.set_yticklabels(month_list, rotation=0, fontsize=22)
#     ax.set_xticks(np.array([0, 7, 14, 21, 28]) * (86400 / time_step))
#     ax.set_xticklabels([0, 7, 14, 21, 28], rotation=0, fontsize=22)
#     ax.set_xlabel('Date', fontsize=25)
#     ax.patch.set_edgecolor('black')
#     ax.patch.set_linewidth(2)
#     ax.set_title(plot_title, fontsize=30)
#     if export_path:
#         plt.savefig(export_path, bbox_inches='tight', transparent=transparent)
#     else:
#         fig.show()
#
#     # Condense timeline figure using probability sum and majority voting
#     matrix_condensed = np.ones((nmonths, matrix_length)) * na_label
#     matrix_pnorm = np.zeros((nmonths, matrix_length))
#     if plot_labels:
#         labeled_matrix_condensed = np.ones((nmonths, matrix_length)) * na_label
#     for i in range(nmonths):
#         for j in range(matrix_length):
#             # Sum probabilities to find best class and store pnorm
#             sub_probs = matrix_probs[nsubrows * i:nsubrows * i + nsubrows, j, :]
#             sub_probs_sum = np.sum(sub_probs, axis=0)
#             sub_probs_contributing_station_count = np.sum(np.sum(sub_probs, axis=1) != 0)
#             matrix_condensed[i, j] = np.argmax(sub_probs_sum) if sub_probs_contributing_station_count != 0 else na_label
#             matrix_pnorm[i, j] = np.max(sub_probs_sum) / sub_probs_contributing_station_count
#             # Use majority voting to condense manual labels
#             if plot_labels:
#                 sub_col = labeled_matrix_plot[nsubrows * i:nsubrows * i + nsubrows, j]
#                 labels_seen, label_counts = np.unique(sub_col, return_counts=True)
#                 if len(labels_seen) == 1 and na_label in labels_seen:
#                     labeled_matrix_condensed[i, j] = na_label
#                 elif len(labels_seen) == 1:
#                     labeled_matrix_condensed[i, j] = labels_seen[0]
#                 else:
#                     if na_label in labels_seen:
#                         label_counts = np.delete(label_counts, labels_seen == na_label)
#                         labels_seen = np.delete(labels_seen, labels_seen == na_label)
#                     selected_label_index = np.argwhere(label_counts == np.amax(label_counts))[-1][0]
#                     labeled_matrix_condensed[i, j] = labels_seen[selected_label_index]
#
#     cbar_kws = {'ticks': real_cbar_ticks,
#                 'drawedges': True,
#                 'aspect': 30}
#
#     fig, ax = plt.subplots(figsize=(20, nmonths))
#     sns.heatmap(matrix_condensed, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.8)
#     if plot_labels:
#         sns.heatmap(labeled_matrix_condensed, cmap=labeled_cmap, cbar=False)
#     cbar = ax.collections[0].colorbar
#     cbar.outline.set_color('black')
#     cbar.outline.set_linewidth(1.5)
#     cbar.ax.set_yticklabels(rgb_keys, fontsize=22)
#     cbar.ax.invert_yaxis()
#     for y in range(0, nmonths):
#         ax.axhline(y=y, color='black')
#     if plot_labels:
#         ax.plot([labeled_start_index, matrix_length], [labeled_start_row, labeled_start_row], 'k-', linewidth=5.5)
#         ax.plot([labeled_start_index, matrix_length], [labeled_start_row + 1, labeled_start_row + 1], 'k-',
#                 linewidth=5.5)
#         for r in range(labeled_start_row, labeled_end_row):
#             ax.plot([0, matrix_length], [(r + 1), (r + 1)], 'k-', linewidth=5.5)
#         ax.plot([0, labeled_end_index], [labeled_end_row, labeled_end_row], 'k-', linewidth=5.5)
#         ax.plot([0, labeled_end_index], [labeled_end_row + 1, labeled_end_row + 1], 'k-', linewidth=5.5)
#     ax.set_yticks(np.arange(0.5, nmonths))
#     ax.set_yticklabels(month_list, rotation=0, fontsize=22)
#     ax.set_xticks(np.array([0, 7, 14, 21, 28]) * (86400 / time_step))
#     ax.set_xticklabels([0, 7, 14, 21, 28], rotation=0, fontsize=22)
#     ax.set_xlabel('Date', fontsize=25)
#     ax.patch.set_edgecolor('black')
#     ax.patch.set_linewidth(2)
#     ax.set_title(plot_title, fontsize=30)
#     if export_path:
#         plt.savefig(export_path[:-4] + '_condensed' + export_path[-4:], bbox_inches='tight', transparent=transparent)
#     else:
#         fig.show()
#     print('Done!')
#
#
#
#
#
#
#

### Calculate uncertainty from model training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

directories = ['/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/',
               '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min_infra/',
               '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min_semi/']

dict1 = {'Broadband\nTremor': 0,
          'Harmonic\nTremor': 1,
          'Monochromatic\nTremor': 2,
          'Non-tremor\nSignal': 3,
          'Explosion': 4,
          'Noise': 5}

dict2 = {'Infrasonic\nTremor': 0,
          'Explosion': 1,
          'Wind Noise': 2,
          'Electronic Noise': 3}

dict3 = {'Broadband\nTremor': 0,
          'Harmonic\nTremor': 1,
          'Monochromatic\nTremor': 2,
          'Non-tremor\nSignal': 3,
          'Long\nPeriod': 4,
          'Explosion': 5,
          'Noise': 6}

dicts = [dict1, dict2, dict3]

for dictx, dir in zip(dicts,directories):
    filepaths = glob.glob(dir + '*.npy')
    class_path_keys = [f.split('_')[-1] for f in filepaths]
    unique_path_keys = list(np.sort(np.unique(class_path_keys)))
    class_counts = [len(glob.glob(dir + '*' + unique_path_key)) for unique_path_key in unique_path_keys]
    ticklabels = [dict((v,k) for k,v in dictx.items())[i] for i in range (len(class_counts))]
    fig, ax = plt.subplots()
    for i, cc in enumerate(class_counts):
        ax.bar(i, cc, color='grey')
        ax.text(i, cc, str(cc), va='bottom', ha='center')
    fig.tight_layout(pad=2.5)
    ax.set_title(dir)
    ax.set_xticks(range(len(class_counts)),ticklabels, fontsize=9)
    ax.set_xlabel('Class')
    ax.set_ylabel('Class Counts')
    ax.grid()
    fig.show()
# fig.savefig('/Users/darrentpk/Desktop/simulations.png',bbox_inches='tight',transparent=False)





########

#
# # Import dependencies
# import glob
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import colorcet as cc
#
# # Define numpy directory
# npy_dir = '/Users/darrentpk/Desktop/PVV_npy/'
#
# # Prepare label dictionary
# label_dict = {'Broadband Tremor': 1,
#               'Harmonic Tremor': 2,
#               'Monochromatic Tremor': 3,
#               'Non-tremor Signal': 4,
#               'Explosion': 5,
#               'Noise': 6}
#
# labels = ['Broadband Tremor', 'Harmonic Tremor', 'Monochromatic Tremor', 'Non-tremor Signal',
#           'Explosion', 'Noise']
#
# # Define conditions
# station_check = 'PVV'
# # PN7A, PS1A, PS4A, PV6A, PVV
#
# for label_check in labels:
#
#     # Filter filenames
#     all_filenames = glob.glob(npy_dir + station_check + '*' + str(label_dict[label_check]) + '.npy')
#
#     # Plot 25
#     N = 16
#
#     # Randomly sample N files from list
#     chosen_filenames = random.sample(all_filenames, N)
#
#     # Craft title
#     fig, axs = plt.subplots(nrows=int(np.sqrt(N)), ncols=int(np.sqrt(N)), figsize=(6,8))
#     fig.suptitle('%d samples of %s slices on %s' % (N, label_check, station_check), fontweight='bold')
#     for i in range(int(np.sqrt(N))):
#         for j in range(int(np.sqrt(N))):
#             filename_index = i * int(np.sqrt(N)) + (j + 1) - 1
#             if filename_index > (len(chosen_filenames)-1):
#                 continue
#             spec_db = np.load(chosen_filenames[filename_index])
#             if np.sum(spec_db < -250) > 0:
#                 print(i, j)
#             axs[i, j].imshow(spec_db, vmin=np.percentile(spec_db, 20), vmax=np.percentile(spec_db, 97.5),
#                            origin='lower', aspect='auto', interpolation=None, cmap=cc.cm.rainbow)
#             axs[i, j].set_xticks([])
#             axs[i, j].set_yticks([])
#     fig.show()

path = '/Users/darrentpk/Desktop/pavlof_chron.csv'
import pandas as pd
df = pd.read_csv(path, header=1)
df = df.fillna(0)

from obspy import UTCDateTime

base_time = UTCDateTime(df.Date[0])
dates =  [UTCDateTime(d) for d in list(df.Date)]
days =   [(d - base_time)/86400 for d in dates]
ash =    [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.Ash==1])]
so2 =    [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.SO2==1])]
t_bar =  [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.Barely==1])]
t_mod =  [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.Moderate==1])]
t_sat =  [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.Saturated==1])]
tremor = [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.TR=='1'])]
lp =     [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.LP==1])]
vlp =    [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.VLP==1])]
eq =     [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.EQ==1])]
ex =     [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.EX==1])]
gca =    [(UTCDateTime(d) - base_time)/86400 for d in list(df.Date[df.GCA==1])]

month_list = []
month_vec = []
month_utcdate = base_time
while month_utcdate <= (UTCDateTime(list(df.Date)[-1])+1):
    month_list.append(month_utcdate.strftime('%b \'%y'))
    month_vec.append(month_utcdate)
    if month_utcdate.month + 1 <= 12:
        month_utcdate += UTCDateTime(month_utcdate.year, month_utcdate.month + 1, 1) - month_utcdate
    else:
        month_utcdate += UTCDateTime(month_utcdate.year + 1, 1, 1) - month_utcdate

total_days = (month_vec[-1]-month_vec[0])/86400
g2y_pct_since = (UTCDateTime(2021,7,9)-base_time)/86400
y2o_pct_since = (UTCDateTime(2021,8,5)-base_time)/86400
o2y_pct_since = (UTCDateTime(2022,12,17)-base_time)/86400
y2g_pct_since = (UTCDateTime(2023,1,19)-base_time)/86400
g2e_pct_since = (month_vec[-1]-base_time)/86400

import numpy as np
import matplotlib.pyplot as plt

start = (UTCDateTime(2021,7,15) - base_time)/86400
end = (UTCDateTime(2021,8,15) - base_time)/86400

fig, ax = plt.subplots(figsize=(10,1.5))
ax.axvspan(0, g2y_pct_since, 1-0.125, 1, color='green')
ax.axvspan(g2y_pct_since, y2o_pct_since, 1-0.125, 1, color='yellow')
ax.axvspan(y2o_pct_since, o2y_pct_since, 1-0.125, 1, color='orange')
ax.axvspan(o2y_pct_since, y2g_pct_since, 1-0.125, 1, color='yellow')
ax.axvspan(y2g_pct_since, g2e_pct_since, 1-0.125, 1, color='green')
ax.plot(ash,np.ones(len(ash))*1,'^',color='grey')
ax.plot(so2,np.ones(len(so2))*2,'.',color='purple')
ax.plot(t_bar,np.ones(len(t_bar))*3,'.',color='gold')
ax.plot(t_mod,np.ones(len(t_mod))*3,'.',color='orange')
ax.plot(t_sat,np.ones(len(t_sat))*3,'.',color='red')
ax.plot(tremor,np.ones(len(tremor))*4,'.',color='grey')
ax.plot(lp,np.ones(len(lp))*5,'.',color='black')
ax.plot(vlp,np.ones(len(vlp))*5,'.',color='black')
ax.plot(eq,np.ones(len(eq))*5,'+',color='red')
ax.plot(ex,np.ones(len(ex))*6,'.',color='maroon')
ax.plot(gca,np.ones(len(gca))*6,'+',color='orange')
ax.set_yticks(range(1,8),['Ash','SO2','Temp','Tremor','EQ','Explosion','AVO'])
ax.set_xticks([(m-base_time)/86400 for m in month_vec],month_list,rotation=30)
ax.set_xlim([start,end])
ax.set_ylim([0.5,7.5])
# fig.savefig('/Users/darrentpk/Desktop/pavlof_chron.png',bbox_inches='tight',dpi=500)
fig.show()


##############


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import seaborn as sns
from obspy import UTCDateTime

# Define time ticks on x-axis
starttime = UTCDateTime(2021,6,1)
endtime = UTCDateTime(2023,10,1)
month_utcdatetimes = []
month_utcdate = starttime
while month_utcdate <= endtime:
    month_utcdatetimes.append(month_utcdate)
    if month_utcdate.month + 1 <= 12:
        month_utcdate += UTCDateTime(month_utcdate.year, month_utcdate.month + 1, 1) - month_utcdate
    else:
        month_utcdate += UTCDateTime(month_utcdate.year + 1, 1, 1) - month_utcdate
xticklabels_horiz = [t.strftime('%b \'%y') for t in month_utcdatetimes]
total_days = (endtime-starttime)/86400
xticks_horiz = [(t - starttime)/86400/total_days for t in month_utcdatetimes]

df = pd.read_csv('./data_frames/semi_df.csv', index_col=0)
df['class'] = df['class'].replace(np.nan,'N/A')
date_list = list(df['time'])
pct_since_arr = np.array([(UTCDateTime(dt) - starttime)/86400/total_days for dt in date_list])
df['pct_since'] = pct_since_arr
rgb_values = np.array([
    [193, 39, 45],
    [0, 129, 118],
    [0, 0, 167],
    [238, 204, 22],
    [103, 72, 132],
    [164, 98, 0],
    [40, 40, 40],
    [255, 255, 255]])
rgb_ratios = rgb_values/255
colors = {'Broadband Tremor': rgb_ratios[0],
          'Harmonic Tremor': rgb_ratios[1],
          'Monochromatic Tremor': rgb_ratios[2],
          'Non-tremor Signal': rgb_ratios[3],
          'Long Period': rgb_ratios[4],
          'Explosion': rgb_ratios[5],
          'Noise': rgb_ratios[6],
          'N/A': rgb_ratios[7]}
hue_order = ['Broadband Tremor',
             'Harmonic Tremor',
             'Monochromatic Tremor',
             'Non-tremor Signal',
             'Long Period',
             'Explosion',
             'Noise',
             'N/A']
listed_colors = np.concatenate((rgb_ratios, np.ones((np.shape(rgb_values)[0],1))), axis=1)
listed_colors = ListedColormap(listed_colors)
class_list = list(df['class'])
class_index = np.array([np.uint8(hue_order.index(c)) for c in class_list])

for ratio in np.arange(0.1,0.7,0.1):
    df2 = df.copy()
    df2['class'][df2['pnorm']<ratio]='Noise'

    multiple = 'fill'
    common_norm = True

    fig, ax = plt.subplots(figsize=(10,3))
    sns.kdeplot(data=df2, x='pct_since', hue='class', hue_order=reversed(hue_order), fill=True,
             palette=colors, linewidth=1, legend=False, ax=ax, common_norm=common_norm, multiple=multiple, bw_method='scott', bw_adjust=0.08)
    # sns.kdeplot(data=df[df['class']=='Broadband Tremor'], x='pct_since', hue='class', hue_order=reversed(hue_order), fill=True,
    #         palette=colors, linewidth=1, legend=False, ax=ax, common_norm=common_norm, multiple=multiple, bw_method='scott', bw_adjust=0.1)
    # sns.kdeplot(data=df[df['class']=='Harmonic Tremor'], x='pct_since', hue='class', hue_order=reversed(hue_order), fill=True,
    #         palette=colors, linewidth=1, legend=False, ax=ax, common_norm=common_norm, multiple=multiple, bw_method='scott', bw_adjust=1.5)
    ax.set_xticks(xticks_horiz)
    ax.set_xticklabels([])
    ax.set(xlabel=None)
    ax.set_xlim([xticks_horiz[0], xticks_horiz[-1]])
    ax_ylim = ax.get_ylim()
    new_ymin = ax_ylim[1] * -0.2
    ax.imshow(class_index.reshape(1,len(class_index)), extent=[0, 1, new_ymin, 0], origin='lower', aspect='auto',
              interpolation='None', cmap=listed_colors, alpha=0.75)
    ax.axhline(y=0, linewidth=2.5, color='k')
    ax.set_ylim([new_ymin, ax_ylim[1]])
    ax.set_ylabel('Seismic\nClass Density',fontsize=15)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xticks(xticks_horiz, xticklabels_horiz, rotation=50)
    fig.tight_layout()
    fig.show()

ax.set_xlim([0.2,0.3])
fig.show()



##########

