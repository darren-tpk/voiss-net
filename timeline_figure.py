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

# Define variables
model_used = 'station-specific'  # or 'station-generic'
stations = ['PN7A', 'PS1A', 'PS4A', 'PV6A', 'PVV']  # ORDER MATTERS
nsubrows = len(stations)
month_list = ['July','August','September','October','November','December','January','February','March','April','May','June']
nmonths = len(month_list)
time_step = 4 * 60  # s
tick_days = [0, 7, 14, 28]

# Make predictions with model
params = {
    "dim": (94, 240),
    "batch_size": 100,
    "n_classes": 6,
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
# matrix_plot = np.random.randint(0, 7, (matrix_height, matrix_length))
matrix_plot = np.ones((matrix_height, matrix_length)) * 6
for indicator in indicators:
    utc = indicator[1]
    if utc < UTCDateTime(2022,7,1):
        row_index = nsubrows * month_list.index(utc.strftime('%B')) + stations.index(indicator[0])
        col_index = int((indicator[1] - UTCDateTime(utc.year,utc.month,1)) / time_step)
        matrix_plot[row_index, col_index] = indicator[2]

# Craft labeled matrix
matrix_plot2 = np.ones((matrix_height, matrix_length)) * 6
labeled_spec_paths = glob.glob('/Users/darrentpk/Desktop/labeled_npy_4min/*.npy')
indicators2 = []
for i, filepath in enumerate(labeled_spec_paths):
    filename = filepath.split('/')[-1]
    chunks = filename.split('_')
    indicators2.append([chunks[0],UTCDateTime(chunks[1]),int(chunks[3][0])-1])
for indicator2 in indicators2:
    utc = indicator2[1]
    if utc < UTCDateTime(2022,7,1):
        row_index = nsubrows * month_list.index(utc.strftime('%B')) + stations.index(indicator2[0])
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
            'drawedges': True,
            'label': 'Classes'}

# Craft timeline figure
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(matrix_plot, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.5)
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
                         'N/A'])
cbar.ax.invert_yaxis()
for y in range(0,matrix_height,5):
    ax.axhline(y=y, color='black')
    indent = 100
    ax.text(indent, y+1, 'PN7A')
    ax.text(indent, y+2, 'PS1A')
    ax.text(indent, y+3, 'PS4A')
    ax.text(indent, y+4, 'PV6A')
    ax.text(indent, y+5, 'PVV')
ax.axhline(y=0, xmin=, xmax=, linewidth=2)
ax.axhline(y=5, xmin=, xmax=, linewidth=2)
ax.set_yticks(np.arange(nsubrows/2,matrix_height,nsubrows))
ax.set_yticklabels(month_list, rotation=0)
ax.set_xticks(np.array([0,7,14,21,28])*(86400/time_step))
ax.set_xticklabels([0,7,14,21,28], rotation=0)
ax.set_xlabel('Date')
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(2)
plt.show()