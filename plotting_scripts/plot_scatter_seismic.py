# Plot scatter plot of properties pertaining to each class

# Import all dependencies
import pandas as pd
import obspy
import numpy as np
from obspy import UTCDateTime
from matplotlib import pyplot as plt

# Read dataframe with all properties
df = pd.read_csv ('scatter_df.csv', usecols=range(1,8))
df['class'] = df['class'].replace(np.nan,'N/A')
df = df.loc[df['class'] != 'Noise']
df = df.loc[df['class'] != 'N/A']

# Define colors
rgb_values = np.array([
    [193,  39,  45],
    [  0, 129, 118],
    [  0,   0, 167],
    [238, 204,  22],
    [164,  98,   0],
    [ 40,  40,  40],
    [255, 255, 255]])
rgb_ratios = rgb_values/255
colors = {'Broadband Tremor': rgb_ratios[0],
          'Harmonic Tremor': rgb_ratios[1],
          'Monochromatic Tremor': rgb_ratios[2],
          'Non-tremor Signal': rgb_ratios[3],
          'Explosion': rgb_ratios[4],
          'Noise': rgb_ratios[5],
          'N/A': rgb_ratios[6]}

# Create scatter color array
cseries = df['class'].map(colors)

# # Generate scatter plot 1 (first column uses dominant frequency)
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].scatter(df['fd'], df['dr'], c=cseries, s=.6, alpha=.3)
# axs[0, 1].scatter(df['fsd'], df['dr'], c=cseries, s=.6, alpha=.3)
# axs[1, 0].scatter(df['fd'], df['pe'], c=cseries, s=.6, alpha=.3)
# axs[1, 1].scatter(df['fsd'], df['pe'], c=cseries, s=.6, alpha=.3)
# axs[0, 0].set_ylabel('$D_R$ (cm$^2$)')
# axs[0, 0].set_xlim([0.9, 2.5])
# axs[0, 0].set_ylim([0, 5])
# axs[0, 0].xaxis.tick_top()
# axs[1, 0].set_ylabel('$p_e$')
# axs[1, 0].set_xlabel('$f_d$ (Hz)')
# axs[1, 0].set_xlim([0.9, 2.5])
# axs[1, 0].set_ylim([0.2, 0.7])
# axs[0, 1].xaxis.tick_top()
# axs[0, 1].yaxis.tick_right()
# axs[0, 1].set_xlim([1, 20])
# axs[0, 1].set_ylim([0, 5])
# axs[1, 1].set_xlabel('$\sigma_f$ (Hz)')
# axs[1, 1].yaxis.tick_right()
# axs[1, 1].set_xlim([1, 20])
# axs[1, 1].set_ylim([0.2, 0.7])
# fig.subplots_adjust(wspace=0.05, hspace=0.05)
# fig.suptitle('Class-wise scatter of calculated metrics')
# fig.show()

# Generate scatter plot 2 (first column uses centroid frequency)
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(df['fc'], df['dr'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[0, 1].scatter(df['fsd'], df['dr'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[1, 0].scatter(df['fc'], df['pe'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[1, 1].scatter(df['fsd'], df['pe'], c=df['class'].map(colors), s=.6, alpha=.3)
axs[0, 0].set_ylabel('$D_R$ (cm$^2$)')
axs[0, 0].set_xlim([2, 7])
axs[0, 0].set_ylim([0, 5])
axs[0, 0].xaxis.tick_top()
axs[1, 0].set_ylabel('$p_e$')
axs[1, 0].set_xlabel('$f_c$ (Hz)')
axs[1, 0].set_xlim([2, 7])
axs[1, 0].set_ylim([0.2, 0.7])
axs[0, 1].xaxis.tick_top()
axs[0, 1].yaxis.tick_right()
axs[0, 1].set_xlim([1, 20])
axs[0, 1].set_ylim([0, 5])
axs[1, 1].set_xlabel('$\sigma_f$ (Hz)')
axs[1, 1].yaxis.tick_right()
axs[1, 1].set_xlim([1, 20])
axs[1, 1].set_ylim([0.2, 0.7])
fig.subplots_adjust(wspace=0.05, hspace=0.05)
fig.suptitle('Class-wise scatter of calculated metrics (Noise Removed)')
fig.show()
#
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].scatter(df['fc'], df['fd'], c=df['class'].map(colors), s=.6, alpha=.3)
# axs[0, 1].scatter(df['fsd'], df['fd'], c=df['class'].map(colors), s=.6, alpha=.3)
# axs[1, 0].scatter(df['fc'], df['pe'], c=df['class'].map(colors), s=.6, alpha=.3)
# axs[1, 1].scatter(df['fsd'], df['pe'], c=df['class'].map(colors), s=.6, alpha=.3)
# axs[0, 0].set_ylabel('$f_d$ (Hz$)')
# axs[0, 0].set_xlim([2, 7])
# axs[0, 0].set_ylim([0.9, 2.5])
# axs[0, 0].xaxis.tick_top()
# axs[1, 0].set_ylabel('$p_e$')
# axs[1, 0].set_xlabel('$f_c$ (Hz)')
# axs[1, 0].set_xlim([2, 7])
# axs[1, 0].set_ylim([0.2, 0.7])
# axs[0, 1].xaxis.tick_top()
# axs[0, 1].yaxis.tick_right()
# axs[0, 1].set_xlim([1, 20])
# axs[0, 1].set_ylim([0.9, 2.5])
# axs[1, 1].set_xlabel('$\sigma_f$ (Hz)')
# axs[1, 1].yaxis.tick_right()
# axs[1, 1].set_xlim([1, 20])
# axs[1, 1].set_ylim([0.2, 0.7])
# fig.subplots_adjust(wspace=0.05, hspace=0.05)
# fig.suptitle('Class-wise scatter of calculated metrics')
# fig.show()

# Now look at histogram plots

# Get booleans pertaining to each class
hcolors = []
bools = []
for key in list(colors.keys()):
    bools.append(df['class'] == key)
    hcolors.append(colors[key])

# Plot one histogram subplot per metric
metrics = list(df.columns)[2:]
metric_lims = [[0, 5], [0.2, 0.7], [2, 7], [0.9, 2.5], [1, 20]]
metric_labels = ['$D_R$ (cm$^2$)', '$p_e$', '$f_c$ (Hz)', '$f_d$ (Hz)', '$\sigma_f$ (Hz)']
fig, axs = plt.subplots(len(metrics), 1, figsize=(6, 8))
for i, m in enumerate(metrics):
    for j in range(len(bools[:-2])):
        if i==0 and j in [3,4]:
            vec = np.array(df[m][bools[j]])
            vec[vec>5] = np.nan
            axs[i].hist(vec, density=True, color=hcolors[j], alpha=0.6, bins=100)
        else:
            axs[i].hist(df[m][bools[j]], density=True, color=hcolors[j], alpha=0.6, bins=100)
    axs[i].set_xlim(metric_lims[i])
    axs[i].set_xlabel(metric_labels[i])
    axs[i].set_ylabel('Density')
fig.subplots_adjust(hspace=0.8)
fig.suptitle('Histogram density plots for all metrics')
fig.show()

#### To create dataframe (retain to use for infrasound)
# # Load both labeled and unlabeled seismic spectrogram classifications
# timeline_classified = np.load('./classifications/pavlof_seismic_unlabeled_classifications.npy')
# timeline_labeled = np.load('./classifications/pavlof_seismic_labeled_classifications.npy')
#
# # Map to actual classes using reverse dictionary
# label_dict = {'Broadband Tremor': 0,
#               'Harmonic Tremor': 1,
#               'Monochromatic Tremor': 2,
#               'Non-tremor Signal': 3,
#               'Explosion': 4,
#               'Noise': 5,
#               'N/A': 6}
# index_dict = {v: k for k, v in label_dict.items()}
# timeline_classified = [index_dict[int(i)] for i in timeline_classified[0]]
# timeline_labeled = [index_dict[int(i)] for i in timeline_labeled[0]]
#
# # Load in corresponding seismic metrics
# metrics_dir = '/Users/darrentpk/Desktop/GitHub/tremor_ml/metrics/PS1A_BHZ_metrics/'
# tag = '_20210101_20230101'
# tmpl = np.load(metrics_dir + 'tmpl_all' + tag + '.npy')
# dr = np.load(metrics_dir + 'dr_all' + tag + '.npy')
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
# dr_retained = []
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
#         dr_retained.append(np.nanmedian(dr[:,ind]))
#         pe_retained.append(np.nanmedian(pe[:,ind]))
#         fc_retained.append(np.nanmedian(fc[:,ind]))
#         fd_retained.append(np.nanmedian(fd[:,ind]))
#         fsd_retained.append(np.nanmedian(fsd[:,ind]))
#     except:
#         dr_retained.append(np.nan)
#         pe_retained.append(np.nan)
#         fc_retained.append(np.nan)
#         fd_retained.append(np.nan)
#         fsd_retained.append(np.nan)
#
# # Create dataframe
# list_of_tuples = list(zip(tutc_classified,timeline_classified,dr_retained,pe_retained,fc_retained,fd_retained,fsd_retained))
# df = pd.DataFrame(list_of_tuples, columns = ['time', 'class', 'dr', 'pe', 'fc', 'fd', 'fsd'])
#