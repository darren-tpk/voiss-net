### To create dataframe (retain to use for infrasound)

# Import all dependencies
import pandas as pd
import obspy
import numpy as np
from obspy import UTCDateTime
from matplotlib import pyplot as plt

# Read dataframe with all properties
df = pd.read_csv ('scatter_df_infra.csv', usecols=range(1,8))
df['class'] = df['class'].replace(np.nan,'N/A')
df = df.loc[df['class'] != 'Noise']
df = df.loc[df['class'] != 'Electronic Noise']
df = df.loc[df['class'] != 'N/A']

# Define colors
rgb_values = np.array([
    [103,  52, 235],
    [235, 152,  52],
    [ 40,  40,  40],
    [ 15,  37,  60],
    [255, 255, 255]])
rgb_ratios = rgb_values/255
colors = {'Infrasonic Tremor': rgb_ratios[0],
          'Explosion': rgb_ratios[1],
          'Noise': rgb_ratios[2],
          'Electronic Noise': rgb_ratios[3],
          'N/A': rgb_ratios[4]}

# Create scatter color array
cseries = df['class'].map(colors)

# Generate scatter plot 1 (first column uses dominant frequency)
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(df['fd'], df['rmsp'], c=cseries, s=.6, alpha=.3)
axs[0, 1].scatter(df['fsd'], df['rmsp'], c=cseries, s=.6, alpha=.3)
axs[1, 0].scatter(df['fd'], df['pe'], c=cseries, s=.6, alpha=.3)
axs[1, 1].scatter(df['fsd'], df['pe'], c=cseries, s=.6, alpha=.3)
axs[0, 0].set_ylabel('$Pa_{ rms}$')
axs[0, 0].set_xlim([0.9, 4])  # fd lims
axs[0, 0].set_ylim([-0.5, 10])  # rmsp lims
axs[0, 0].xaxis.tick_top()
axs[1, 0].set_ylabel('$p_e$')
axs[1, 0].set_xlabel('$f_d$ (Hz)')
axs[1, 0].set_xlim([0.9, 4])  # fd lims
axs[1, 0].set_ylim([0.5, 0.7])  # pe lims
axs[0, 1].xaxis.tick_top()
axs[0, 1].yaxis.tick_right()
axs[0, 1].set_xlim([0, 8])  # fsd lims
axs[0, 1].set_ylim([-0.5, 10])  # rmsp lims 
axs[1, 1].set_xlabel('$\sigma_f$ (Hz)')
axs[1, 1].yaxis.tick_right()
axs[1, 1].set_xlim([0, 8])  # fsd lims
axs[1, 1].set_ylim([0.5, 0.7])  # pe lims
fig.subplots_adjust(wspace=0.05, hspace=0.05)
fig.suptitle('Class-wise scatter of calculated metrics (Noise Removed)')
fig.show()

# Generate scatter plot 2 (first column uses centroid frequency)
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(df['fc'], df['rmsp'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[0, 1].scatter(df['fsd'], df['rmsp'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[1, 0].scatter(df['fc'], df['pe'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[1, 1].scatter(df['fsd'], df['pe'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[0, 0].set_ylabel('$Pa_{ rms}$')
axs[0, 0].set_xlim([3, 6])  # fc lims
axs[0, 0].set_ylim([-0.5, 10])  # rmsp lims
axs[0, 0].xaxis.tick_top()
axs[1, 0].set_ylabel('$p_e$')
axs[1, 0].set_xlabel('$f_c$ (Hz)')
axs[1, 0].set_xlim([3, 6])  # fc lims
axs[1, 0].set_ylim([0.5, 0.7])  # pe lims
axs[0, 1].xaxis.tick_top()
axs[0, 1].yaxis.tick_right()
axs[0, 1].set_xlim([0, 8])  # fsd lims
axs[0, 1].set_ylim([-0.5, 10])  # rmsp lims
axs[1, 1].set_xlabel('$\sigma_f$ (Hz)')
axs[1, 1].yaxis.tick_right()
axs[1, 1].set_xlim([0, 8])  # fsd lims
axs[1, 1].set_ylim([0.5, 0.7])  # pe lims
fig.subplots_adjust(wspace=0.05, hspace=0.05)
fig.suptitle('Class-wise scatter of calculated metrics (Noise Removed)')
fig.show()

fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(df['fc'], df['fd'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[0, 1].scatter(df['fsd'], df['fd'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[1, 0].scatter(df['fc'], df['pe'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[1, 1].scatter(df['fsd'], df['pe'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[0, 0].set_ylabel('$f_d$ (Hz)')
axs[0, 0].set_xlim([3, 6])  # fc lims
axs[0, 0].set_ylim([0.9, 4])  # fd lims
axs[0, 0].xaxis.tick_top()
axs[1, 0].set_ylabel('$p_e$')
axs[1, 0].set_xlabel('$f_c$ (Hz)')
axs[1, 0].set_xlim([3, 6])  # fc lims
axs[1, 0].set_ylim([0.5, 0.7])  # pe lims
axs[0, 1].xaxis.tick_top()
axs[0, 1].yaxis.tick_right()
axs[0, 1].set_xlim([0, 8])  # fsd lims
axs[0, 1].set_ylim([0.9, 4])  # fd lims
axs[1, 1].set_xlabel('$\sigma_f$ (Hz)')
axs[1, 1].yaxis.tick_right()
axs[1, 1].set_xlim([0, 8])  # fsd lims
axs[1, 1].set_ylim([0.5, 0.7])  # pe lims
fig.subplots_adjust(wspace=0.05, hspace=0.05)
fig.suptitle('Class-wise scatter of calculated metrics (Noise Removed)')
fig.show()

# Now look at histogram plots

# Get booleans pertaining to each class
hcolors = []
bools = []
for key in list(colors.keys()):
    bools.append(df['class'] == key)
    hcolors.append(colors[key])

# Plot one histogram subplot per metric
metrics = list(df.columns)[2:]
metric_lims = [[0, 10], [0.5, 0.7], [3, 6], [0.9, 4], [0, 8]]
metric_labels = ['$Pa_{ rms}$', '$p_e$', '$f_c$ (Hz)', '$f_d$ (Hz)', '$\sigma_f$ (Hz)']
fig, axs = plt.subplots(len(metrics), 1, figsize=(6, 8))
for i, m in enumerate(metrics):
    for j in range(len(bools[:-2])):
        if i==0 and j in [3,4]:
            vec = np.array(df[m][bools[j]])
            vec[vec>5] = np.nan
            axs[i].hist(vec, density=True, color=hcolors[j], alpha=0.6, bins=200)
        else:
            axs[i].hist(df[m][bools[j]], density=True, color=hcolors[j], alpha=0.6, bins=100)
    axs[i].set_xlim(metric_lims[i])
    axs[i].set_xlabel(metric_labels[i])
    axs[i].set_ylabel('Density')
fig.subplots_adjust(hspace=0.8)
fig.suptitle('Histogram density plots for all metrics')
fig.show()

# # Load both labeled and unlabeled seismic spectrogram classifications
# import numpy as np
# from obspy import UTCDateTime
# import pandas as pd
# timeline_classified = np.load('./classifications/pavlof_infra_unlabeled_classifications.npy')
# timeline_labeled = np.load('./classifications/pavlof_infra_labeled_classifications.npy')
#
# # Map to actual classes using reverse dictionary
# label_dict = {'Infrasonic Tremor': 0,
#               'Explosion': 1,
#               'Noise': 2,
#               'Electronic Noise': 3,
#               'N/A': 4}
# index_dict = {v: k for k, v in label_dict.items()}
# timeline_classified = [index_dict[int(i)] for i in timeline_classified[0]]
# timeline_labeled = [index_dict[int(i)] for i in timeline_labeled[0]]
#
# # Load in corresponding seismic metrics
# metrics_dir = '/Users/darrentpk/Desktop/GitHub/tremor_ml/metrics/PS4A_BDF_metrics/'
# tag = '_20210101_20230101'
# tmpl = np.load(metrics_dir + 'tmpl_all' + tag + '.npy')
# rmsp = np.load(metrics_dir + 'rmsp_all' + tag + '.npy')
# pe = np.load(metrics_dir + 'pe_all' + tag + '.npy')
# fc = np.load(metrics_dir + 'fc_all' + tag + '.npy')
# fd = np.load(metrics_dir + 'fd_all' + tag + '.npy')
# fsd = np.load(metrics_dir + 'fsd_all' + tag + '.npy')
# spec_db = np.load(metrics_dir + 'spec_db_all' + tag + '.npy')
#
# # Convert matplotlib times to match with timeline_classified more easily
# tutc = []
# for t in tmpl[0]:
#     if np.isnan(t):
#         tutc.append(UTCDateTime(1970,1,1))
#     else:
#         tutc.append(UTCDateTime(1970,1,1) + t*86400)
# tutc = np.array(tutc)
#
# # Create corresponding time vector for timeline_classified
# tutc_classified = np.arange(UTCDateTime(2021,1,1,0,2),UTCDateTime(2023,1,1),240)
# tutc_classified = list(tutc_classified)
# rmsp_retained = []
# pe_retained = []
# fc_retained = []
# fd_retained = []
# fsd_retained = []
#
# test_tutc = list(tutc - .008)
#
# for i, t in enumerate(tutc_classified):
#     print(i)
#     try:
#         ind = test_tutc.index(t)
#         rmsp_retained.append(np.nanmedian(rmsp[:,ind]))
#         pe_retained.append(np.nanmedian(pe[:,ind]))
#         fc_retained.append(np.nanmedian(fc[:,ind]))
#         fd_retained.append(np.nanmedian(fd[:,ind]))
#         fsd_retained.append(np.nanmedian(fsd[:,ind]))
#     except:
#         rmsp_retained.append(np.nan)
#         pe_retained.append(np.nan)
#         fc_retained.append(np.nan)
#         fd_retained.append(np.nan)
#         fsd_retained.append(np.nan)
#
# # Create dataframe
# list_of_tuples = list(zip(tutc_classified,timeline_classified,rmsp_retained,pe_retained,fc_retained,fd_retained,fsd_retained))
# df = pd.DataFrame(list_of_tuples, columns = ['time', 'class', 'rmsp', 'pe', 'fc', 'fd', 'fsd'])
#
# df.to_csv('scatter_df_infra.csv')