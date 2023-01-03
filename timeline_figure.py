# %% CREATE TIMELINE FIGURE

# Import all dependencies
import json
from obspy import UTCDateTime
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from DataGenerator import DataGenerator
from keras.models import load_model
from scipy import stats

# Define variables
model_used = 'station-specific'  # or 'station-generic'
stations = ['PN7A', 'PS1A', 'PS4A', 'PV6A', 'PVV']  # ORDER MATTERS
nsubrows = len(stations)
month_list = ['Jul 21', 'Aug 21', 'Sep 21', 'Oct 21', 'Nov 21', 'Dec 21','Jan 22', 'Feb 22', 'Mar 22', 'Apr 22', 'May 22', 'Jun 22','Jul 22', 'Aug 22', 'Sep 22', 'Oct 22', 'Nov 22', 'Dec 22']
nmonths = len(month_list)
nclasses = 6
na_label = nclasses
time_step = 4 * 60  # s
tick_days = [0, 7, 14, 28]
labeled_start = UTCDateTime(2021, 7, 22)
labeled_end = UTCDateTime(2021, 8, 19)
labeled_start_index = int((labeled_start - UTCDateTime(labeled_start.year, labeled_start.month, 1))/time_step)
labeled_end_index = int((labeled_end - UTCDateTime(labeled_end.year, labeled_end.month, 1))/time_step)

# Make predictions with model
params = {
    "dim": (94, 240),
    "batch_size": 100,
    "n_classes": nclasses,
    "shuffle": True,
}
if model_used == 'station-generic':
    spec_paths = glob.glob('/Users/darrentpk/Desktop/all_npy/*.npy')
    spec_placeholder_labels = [0 for i in spec_paths]
    spec_label_dict = dict(zip(spec_paths, spec_placeholder_labels))
    spec_gen = DataGenerator(spec_paths, spec_label_dict, **params)
    saved_model = load_model('/Users/darrentpk/Desktop/GitHub/tremor_ml/models/4min_all_model.h5')
    spec_predictions = saved_model.predict(spec_gen)
    predicted_labels = np.argmax(spec_predictions, axis=1)  # why are the lengths different?
    indicators = []
    for i, filepath in enumerate(spec_gen.list_ids):
        filename = filepath.split('/')[-1]
        chunks = filename.split('_')
        indicators.append([chunks[0],UTCDateTime(chunks[1]),predicted_labels[i]])
elif model_used == 'station-specific':
    indicators = []
    for station in stations:
        spec_paths = glob.glob('/Users/darrentpk/Desktop/all_npy/*' + station + '*.npy')
        spec_placeholder_labels = [0 for i in spec_paths]
        spec_label_dict = dict(zip(spec_paths, spec_placeholder_labels))
        spec_gen = DataGenerator(spec_paths, spec_label_dict, **params)
        saved_model = load_model('/Users/darrentpk/Desktop/GitHub/tremor_ml/models/4min_' + station +'_model.h5')
        spec_predictions = saved_model.predict(spec_gen)
        predicted_labels = np.argmax(spec_predictions, axis=1)  # why are the lengths different?
        for i, filepath in enumerate(spec_gen.list_ids):
            filename = filepath.split('/')[-1]
            chunks = filename.split('_')
            indicators.append([chunks[0], UTCDateTime(chunks[1]), predicted_labels[i]])

# Craft unlabeled matrix
matrix_length = int(31 * (86400/time_step))
matrix_height = nmonths * nsubrows
matrix_plot = np.ones((matrix_height, matrix_length)) * na_label
for indicator in indicators:
    utc = indicator[1]
    row_index = nsubrows * month_list.index(utc.strftime('%b %y')) + stations.index(indicator[0])
    col_index = int((indicator[1] - UTCDateTime(utc.year,utc.month,1)) / time_step)
    matrix_plot[row_index, col_index] = indicator[2]

# Craft labeled matrix
matrix_plot2 = np.ones((matrix_height, matrix_length)) * na_label
labeled_spec_paths = glob.glob('/Users/darrentpk/Desktop/labeled_npy_4min_2/*.npy')
indicators2 = []
for i, filepath in enumerate(labeled_spec_paths):
    filename = filepath.split('/')[-1]
    chunks = filename.split('_')
    indicators2.append([chunks[0],UTCDateTime(chunks[1]),int(chunks[3][0])-1])
for indicator2 in indicators2:
    utc = indicator2[1]
    row_index = nsubrows * month_list.index(utc.strftime('%b %y')) + stations.index(indicator2[0])
    col_index = int((indicator2[1] - UTCDateTime(utc.year,utc.month,1)) / time_step)
    matrix_plot2[row_index, col_index] = indicator2[2]

# Craft color maps
rgb_values = np.array([
    [255,  13,   0],
    [251,   0, 255],
    [  4,   0, 255],
    [255, 200,   0],
    [  3, 204,   0],
    [  0,   0,   0],
    [255, 255, 255]])
rgb_ratios = rgb_values/255
colors = np.concatenate((rgb_ratios, np.ones((np.shape(rgb_values)[0],1))), axis=1)
cmap = ListedColormap(colors)
colors[-1][-1] = 0
cmap2 = ListedColormap(colors)

# Colorbar keywords
real_cbar_tick_interval = 2 * (len(np.unique(matrix_plot))-1)/(2*np.shape(rgb_values)[0])
real_cbar_ticks = np.arange(real_cbar_tick_interval/2,len(np.unique(matrix_plot))-1,real_cbar_tick_interval)
cbar_kws = {'ticks': real_cbar_ticks,
            'drawedges': True}
            #'label': 'Classes'}

# Craft timeline figure
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(matrix_plot, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.75)
sns.heatmap(matrix_plot2, cmap=cmap2, cbar=False)
cbar = ax.collections[0].colorbar
cbar.outline.set_color('black')
cbar.outline.set_linewidth(1.5)
cbar.ax.set_yticklabels(['Broadband\nTremor',
                         'Harmonic\nTremor',
                         'Monochromatic\nTremor',
                         'Non-Tremor\nSignal',
                         'Explosion',
                         'Noise',
                         'N/A'], fontsize=22)
cbar.ax.invert_yaxis()
for y in range(0,matrix_height,5):
    ax.axhline(y=y, color='black')
    indent = 100
    ax.text(indent, y+1, 'PN7A', fontsize=15)
    ax.text(indent, y+2, 'PS1A', fontsize=15)
    ax.text(indent, y+3, 'PS4A', fontsize=15)
    ax.text(indent, y+4, 'PV6A', fontsize=15)
    ax.text(indent, y+5, 'PVV', fontsize=15)
ax.plot([labeled_start_index, matrix_length], [0, 0], 'k-', linewidth=8.5)
ax.plot([labeled_start_index, matrix_length], [5, 5], 'k-', linewidth=5.5)
ax.plot([0, labeled_end_index], [5, 5], 'k-', linewidth=5.5)
ax.plot([0, labeled_end_index], [10, 10], 'k-', linewidth=5.5)
ax.set_yticks(np.arange(nsubrows/2,matrix_height,nsubrows))
ax.set_yticklabels(month_list, rotation=0, fontsize=22)
ax.set_xticks(np.array([0, 7, 14, 21, 28])*(86400/time_step))
ax.set_xticklabels([0, 7,14, 21 , 28], rotation=0, fontsize=22)
ax.set_xlabel('Date', fontsize=25)
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(2)
ax.set_title('Timeline for Pavlof Seismic Tremor (2021-07-01 to 2022-12-31)', fontsize=30)
plt.savefig('/Users/darrentpk/Desktop/timeline4.pdf')
plt.show()

# Condense timeline figure using station-based voting
matrix_condensed = np.ones((nmonths, matrix_length)) * na_label  # nclasses = N/A class
matrix_condensed2 = np.ones((nmonths, matrix_length)) * na_label
for i in range(nmonths):
    for j in range(matrix_length):
        # first do it for matrix_plot
        sub_col = matrix_plot[5*i:5*i+5, j]
        labels_seen, label_counts = np.unique(sub_col, return_counts=True)
        if len(labels_seen) == 1 and na_label in labels_seen:
            matrix_condensed[i, j] = na_label
        elif len(labels_seen) == 1:
            matrix_condensed[i,j] = labels_seen[0]
        else:
            if na_label in labels_seen:
                label_counts = np.delete(label_counts, labels_seen==na_label)
                labels_seen = np.delete(labels_seen, labels_seen==na_label)
            selected_label_index = np.argwhere(label_counts == np.amax(label_counts))[-1][0]
            # selected_label_index = np.argmax(label_counts)
            matrix_condensed[i, j] = labels_seen[selected_label_index]
        # now do it for matrix_plot2
        sub_col2 = matrix_plot2[5*i:5*i+5, j]
        labels_seen2, label_counts2 = np.unique(sub_col2, return_counts=True)
        if len(labels_seen2) == 1 and na_label in labels_seen2:
            matrix_condensed2[i, j] = na_label
        elif len(labels_seen2) == 1:
            matrix_condensed2[i, j] = labels_seen2[0]
        else:
            if na_label in labels_seen2:
                label_counts2 = np.delete(label_counts2, labels_seen2 == na_label)
                labels_seen2 = np.delete(labels_seen2, labels_seen2 == na_label)
            selected_label_index = np.argwhere(label_counts2 == np.amax(label_counts2))[-1][0]
            # selected_label_index = np.argmax(label_counts2)
            matrix_condensed2[i, j] = labels_seen2[selected_label_index]

fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(matrix_condensed, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.75)
sns.heatmap(matrix_condensed2, cmap=cmap2, cbar=False)
cbar = ax.collections[0].colorbar
cbar.outline.set_color('black')
cbar.outline.set_linewidth(1.5)
cbar.ax.set_yticklabels(['Broadband\nTremor',
                         'Harmonic\nTremor',
                         'Monochromatic\nTremor',
                         'Non-Tremor\nSignal',
                         'Explosion',
                         'Noise',
                         'N/A'], fontsize=22)
cbar.ax.invert_yaxis()
for y in range(0,nmonths):
    ax.axhline(y=y, color='black')
ax.plot([labeled_start_index, matrix_length], [0, 0], 'k-', linewidth=8.5)
ax.plot([labeled_start_index, matrix_length], [1, 1], 'k-', linewidth=5.5)
ax.plot([0, labeled_end_index], [1, 1], 'k-', linewidth=5.5)
ax.plot([0, labeled_end_index], [2, 2], 'k-', linewidth=5.5)
ax.set_yticks(np.arange(0.5,nmonths))
ax.set_yticklabels(month_list, rotation=0, fontsize=22)
ax.set_xticks(np.array([0, 7, 14, 21, 28])*(86400/time_step))
ax.set_xticklabels([0, 7,14, 21 , 28], rotation=0, fontsize=22)
ax.set_xlabel('Date', fontsize=25)
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(2)
ax.set_title('Timeline for Pavlof Seismic Tremor (2021-07-01 to 2022-12-31)', fontsize=30)
plt.savefig('/Users/darrentpk/Desktop/timeline5.pdf')
plt.show()
