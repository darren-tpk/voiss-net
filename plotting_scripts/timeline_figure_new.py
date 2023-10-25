# %% CREATE TIMELINE FIGURE

# Import all dependencies
from obspy import UTCDateTime
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from DataGenerator import DataGenerator
from keras.models import load_model

# Arguments
type = 'seismic'
start_month = UTCDateTime(2021, 1, 1)
end_month = UTCDateTime(2023, 3, 1)
time_step = 60
model_path = './models/4min_all_augmented_drop_model.h5'
meanvar_path = './models/4min_all_augmented_drop_meanvar.npy'
npy_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2021_2023_npy/'
plot_title = 'Timeline for Pavlof Seismic Tremor (2021-01-01 to 2022-12-31)'
figsize = (20,20)
export_path = '/Users/darrentpk/Desktop/test2.png'
transparent = False
plot_labels = False
labels_kwargs = {'start_date': UTCDateTime(2021, 7, 22),
                 'end_date': UTCDateTime(2021, 9, 22),
                 'labels_dir': '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/'}

# Arguments
type = 'seismic'
start_month = UTCDateTime(2014,8,13)
end_month = UTCDateTime(2015,2,26)
time_step = 240
model_path = './models/4min_all_augmented_drop_model.h5'
meanvar_path = './models/4min_all_augmented_drop_meanvar.npy'
npy_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2014_npy_2/'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (start_month.strftime('%Y-%m-%d'),end_month.strftime('%Y-%m-%d'))
figsize = (20,20)
export_path = '/Users/darrentpk/Desktop/pavlof2014.png'
transparent = False
plot_labels = False
labels_kwargs = None

# Filter spec paths by time
spec_paths_full = glob.glob(npy_dir + '*.npy')
spec_paths = []
for spec_path in spec_paths_full:
    utc = UTCDateTime(spec_path.split('/')[-1].split('_')[1])
    spec_station = spec_path.split('/')[-1].split('_')[0]
    if start_month <= utc < end_month:
        spec_paths.append(spec_path)

# Determine number of stations and number of months
spec_path = glob.glob(npy_dir + '*.npy')
stations = list(np.unique([(f.split('/')[-1]).split('_')[0] for f in spec_path]))
nsubrows = len(stations)
month_list = []
month_utcdate = start_month
while month_utcdate < end_month:
    month_list.append(month_utcdate.strftime('%b \'%y'))
    month_utcdate += 31*86400
    month_utcdate = UTCDateTime(month_utcdate.year,month_utcdate.month,1)
nmonths = len(month_list)

# Load model to determine number of classes
saved_model = load_model(model_path)
nclasses = saved_model.layers[-1].get_config()['units']
na_label = nclasses

# Extract mean and variance from training
saved_meanvar = np.load(meanvar_path)
running_x_mean = saved_meanvar[0]
running_x_var = saved_meanvar[1]

# Define fixed params
TICK_DAYS = [0, 7, 14, 28]

# Create data generator for input spec paths
if len(spec_paths) > 2048:
    from functools import reduce
    factors = np.array(reduce(list.__add__,
                              ([i, len(spec_paths) // i] for i in
                               range(1, int(len(spec_paths) ** 0.5) + 1) if
                               len(spec_paths) % i == 0)))
    batch_size = np.max(factors[factors <= 2048])
else:
    batch_size = len(spec_paths)

params = {
    "dim": (saved_model.input_shape[1], saved_model.input_shape[2]),
    "batch_size": batch_size,
    "n_classes": nclasses,
    "shuffle": False,
}
spec_placeholder_labels = [0 for i in spec_paths]
spec_label_dict = dict(zip(spec_paths, spec_placeholder_labels))
spec_gen = DataGenerator(spec_paths, spec_label_dict, **params, is_training=False,
                         running_x_mean=running_x_mean, running_x_var=running_x_var)

# Make predictions
spec_predictions = saved_model.predict(spec_gen)
predicted_labels = np.argmax(spec_predictions, axis=1)
indicators = []
for i, filepath in enumerate(spec_gen.list_ids):
    filename = filepath.split('/')[-1]
    chunks = filename.split('_')
    indicators.append([chunks[0], UTCDateTime(chunks[1]) + int(np.round(saved_model.input_shape[2] / 2)),
                       predicted_labels[i], spec_predictions[i, :]])

# Craft unlabeled matrix and store probabilities
matrix_length = int(31 * (86400/time_step))
matrix_height = nmonths * nsubrows
matrix_plot = np.ones((matrix_height, matrix_length)) * na_label
matrix_probs = np.zeros((matrix_height, matrix_length, nclasses))
for indicator in indicators:
    utc = indicator[1]
    row_index = nsubrows * month_list.index(utc.strftime('%b \'%y')) + stations.index(indicator[0])
    col_index = int((indicator[1] - UTCDateTime(utc.year,utc.month,1)) / time_step)
    matrix_plot[row_index, col_index] = indicator[2]
    matrix_probs[row_index, col_index, :] = indicator[3]

# Craft labeled matrix
if plot_labels:
    labeled_start_index = int((labels_kwargs['start_date'] - UTCDateTime(labels_kwargs['start_date'].year, labels_kwargs['start_date'].month, 1))/time_step)
    labeled_end_index = int((labels_kwargs['end_date'] - UTCDateTime(labels_kwargs['end_date'].year, labels_kwargs['end_date'].month, 1))/time_step)
    labeled_start_row = int(np.floor((labels_kwargs['start_date'] - start_month) / (31*86400)))
    labeled_end_row = int(np.floor((labels_kwargs['end_date'] - start_month) / (31*86400)))
    labeled_matrix_plot = np.ones((matrix_height, matrix_length)) * na_label
    labeled_spec_paths = glob.glob(labels_kwargs['labels_dir'] + '*.npy')
    labeled_indicators = []
    for i, filepath in enumerate(labeled_spec_paths):
        filename = filepath.split('/')[-1]
        chunks = filename.split('_')
        labeled_indicators.append([chunks[0],UTCDateTime(chunks[1]),int(chunks[3][0])])
    for labeled_indicator in labeled_indicators:
        utc = labeled_indicator[1]
        row_index = nsubrows * month_list.index(utc.strftime('%b \'%y')) + stations.index(labeled_indicator[0])
        col_index = int((labeled_indicator[1] - UTCDateTime(utc.year,utc.month,1)) / time_step)
        labeled_matrix_plot[row_index, col_index:col_index+int(240/time_step)] = labeled_indicator[2]

# Choose color palette depending on data type and nclasses
if type == 'seismic':
    # Craft corresponding rgb values
    if nclasses == 6:
        rgb_values = np.array([
            [193, 39, 45],
            [0, 129, 118],
            [0, 0, 167],
            [238, 204, 22],
            [164, 98, 0],
            [40, 40, 40],
            [255, 255, 255]])
        rgb_keys = ['Broadband\nTremor',
                    'Harmonic\nTremor',
                    'Monochromatic\nTremor',
                    'Non-Tremor\nSignal',
                    'Explosion',
                    'Noise',
                    'N/A']
    elif nclasses == 7:
        rgb_values = np.array([
            [193, 39, 45],
            [0, 129, 118],
            [0, 0, 167],
            [238, 204, 22],
            [103, 72, 132],
            [164, 98, 0],
            [40, 40, 40],
            [255, 255, 255]])
        rgb_keys = ['Broadband\nTremor',
                    'Harmonic\nTremor',
                    'Monochromatic\nTremor',
                    'Non-Tremor\nSignal',
                    'Long\nPeriod',
                    'Explosion',
                    'Noise',
                    'N/A']
else:
    # Craft corresponding rgb values
    rgb_values = np.array([
        [103, 52, 235],
        [235, 152, 52],
        [40, 40, 40],
        [15, 37, 60],
        [255, 255, 255]])
    rgb_keys = ['Infrasonic\nTremor',
                'Explosion',
                'Wind\nNoise',
                'Electronic\nNoise',
                'N/A']
rgb_ratios = rgb_values/255
colors = np.concatenate((rgb_ratios, np.ones((np.shape(rgb_values)[0],1))), axis=1)
cmap = ListedColormap(colors)
if plot_labels:
    colors[-1][-1] = 0
    labeled_cmap = ListedColormap(colors)

# Colorbar keywords
real_cbar_tick_interval = 2 * (len(np.unique(matrix_plot))-1)/(2*np.shape(rgb_values)[0])
real_cbar_ticks = np.arange(real_cbar_tick_interval/2,len(np.unique(matrix_plot))-1,real_cbar_tick_interval)
cbar_kws = {'ticks': real_cbar_ticks,
            'drawedges': True,
            'aspect': 30}

# Craft timeline figure
fig, ax = plt.subplots(figsize=(20,nmonths*nsubrows/3))
sns.heatmap(matrix_plot, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.8, vmin=0, vmax=nclasses)
if plot_labels:
    sns.heatmap(labeled_matrix_plot, cmap=labeled_cmap, cbar=False)
cbar = ax.collections[0].colorbar
cbar.outline.set_color('black')
cbar.outline.set_linewidth(1.5)
cbar.ax.set_yticklabels(rgb_keys, fontsize=22)
cbar.ax.invert_yaxis()
for y in range(0,matrix_height,nsubrows):
    ax.axhline(y=y, color='black')
    for i, station in enumerate(stations):
        ax.text(100, y+i+1, station, fontsize=15)
if plot_labels:
    ax.plot([labeled_start_index, matrix_length], [labeled_start_row*nsubrows, labeled_start_row*nsubrows], 'k-', linewidth=5.5)
    ax.plot([labeled_start_index, matrix_length], [(labeled_start_row+1)*nsubrows, (labeled_start_row+1)*nsubrows], 'k-', linewidth=5.5)
    for r in range(labeled_start_row,labeled_end_row):
        ax.plot([0, matrix_length], [(r+1)*nsubrows, (r+1)*nsubrows], 'k-', linewidth=5.5)
    ax.plot([0, labeled_end_index], [labeled_end_row*nsubrows, labeled_end_row*nsubrows], 'k-', linewidth=5.5)
    ax.plot([0, labeled_end_index], [(labeled_end_row+1)*nsubrows, (labeled_end_row+1)*nsubrows], 'k-', linewidth=5.5)
ax.set_yticks(np.arange(nsubrows/2,matrix_height,nsubrows))
ax.set_yticklabels(month_list, rotation=0, fontsize=22)
ax.set_xticks(np.array([0, 7, 14, 21, 28])*(86400/time_step))
ax.set_xticklabels([0, 7, 14, 21 , 28], rotation=0, fontsize=22)
ax.set_xlabel('Date', fontsize=25)
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(2)
ax.set_title(plot_title, fontsize=30)
if export_path:
    plt.savefig(export_path, bbox_inches='tight', transparent=transparent)
else:
    fig.show()

# Condense timeline figure using probability sum and majority voting
matrix_condensed = np.ones((nmonths, matrix_length)) * na_label
matrix_pnorm = np.zeros((nmonths, matrix_length))
if plot_labels:
    labeled_matrix_condensed = np.ones((nmonths, matrix_length)) * na_label
for i in range(nmonths):
    for j in range(matrix_length):
        # Sum probabilities to find best class and store pnorm
        sub_probs = matrix_probs[nsubrows*i:nsubrows*i+nsubrows, j, :]
        sub_probs_sum = np.sum(sub_probs, axis=0)
        sub_probs_contributing_station_count = np.sum(np.sum(sub_probs, axis=1) != 0)
        matrix_condensed[i, j] = np.argmax(sub_probs_sum) if sub_probs_contributing_station_count != 0 else na_label
        matrix_pnorm[i, j] = np.max(sub_probs_sum)/sub_probs_contributing_station_count
        # Use majority voting to condense manual labels
        if plot_labels:
            sub_col = labeled_matrix_plot[nsubrows*i:nsubrows*i+nsubrows, j]
            labels_seen, label_counts = np.unique(sub_col, return_counts=True)
            if len(labels_seen) == 1 and na_label in labels_seen:
                labeled_matrix_condensed[i, j] = na_label
            elif len(labels_seen) == 1:
                labeled_matrix_condensed[i, j] = labels_seen[0]
            else:
                if na_label in labels_seen:
                    label_counts = np.delete(label_counts, labels_seen == na_label)
                    labels_seen = np.delete(labels_seen, labels_seen == na_label)
                selected_label_index = np.argwhere(label_counts == np.amax(label_counts))[-1][0]
                labeled_matrix_condensed[i, j] = labels_seen[selected_label_index]

cbar_kws = {'ticks': real_cbar_ticks,
            'drawedges': True,
            'aspect': 30}

fig, ax = plt.subplots(figsize=(20,nmonths))
sns.heatmap(matrix_condensed, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.8)
if plot_labels:
    sns.heatmap(labeled_matrix_condensed, cmap=labeled_cmap, cbar=False)
cbar = ax.collections[0].colorbar
cbar.outline.set_color('black')
cbar.outline.set_linewidth(1.5)
cbar.ax.set_yticklabels(rgb_keys, fontsize=22)
cbar.ax.invert_yaxis()
for y in range(0,nmonths):
    ax.axhline(y=y, color='black')
if plot_labels:
    ax.plot([labeled_start_index, matrix_length], [labeled_start_row, labeled_start_row], 'k-', linewidth=5.5)
    ax.plot([labeled_start_index, matrix_length], [labeled_start_row+1, labeled_start_row+1], 'k-', linewidth=5.5)
    for r in range(labeled_start_row,labeled_end_row):
        ax.plot([0, matrix_length], [(r+1), (r+1)], 'k-', linewidth=5.5)
    ax.plot([0, labeled_end_index], [labeled_end_row, labeled_end_row], 'k-', linewidth=5.5)
    ax.plot([0, labeled_end_index], [labeled_end_row+1, labeled_end_row+1], 'k-', linewidth=5.5)
ax.set_yticks(np.arange(0.5,nmonths))
ax.set_yticklabels(month_list, rotation=0, fontsize=22)
ax.set_xticks(np.array([0, 7, 14, 21, 28])*(86400/time_step))
ax.set_xticklabels([0, 7,14, 21 , 28], rotation=0, fontsize=22)
ax.set_xlabel('Date', fontsize=25)
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(2)
ax.set_title('Timeline for Pavlof Seismic Tremor (2021/01/01 to 2022/12/31)', fontsize=30)
if export_path:
    plt.savefig(export_path[:-4] + '_condensed' + export_path[-4:], bbox_inches='tight', transparent=transparent)
else:
    fig.show()

