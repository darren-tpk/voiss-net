# %% CHECK TIMELINE PREDICTIONS

# Import all dependencies
from obspy import UTCDateTime
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from DataGenerator import DataGenerator
from keras.models import load_model
from waveform_collection import gather_waveforms
from toolbox import process_waveform, plot_spectrogram_multi, compute_metrics

# Transfer model details
balance_training = True
balance_type = 'subsampled2'  # 'oversampled', 'undersampled', or 'subsampled[0,1,2,3,4,5]'

# Define variables
model_used = 'station-generic'  # or 'station-generic'
# stations = ['PN7A','PS1A', 'PS4A', 'PV6', 'PVV']  # ORDER MATTERS
# stations = ['PS1A', 'PS4A', 'PVV']
stations = ['ISLZ','ISNN','SSBA','SSLN','SSLS']
target_npy_directory = 'shishaldin_2019_2020_npy'
nsubrows = len(stations)
nclasses = 6
time_step = 4 * 60  # s
nhours = 2
starttime = UTCDateTime(2020, 4, 23, 17, 0, 0)
endtime = starttime + nhours*3600
plot_spectrogram = True
plot_metrics = False

# Make predictions with model
params = {
    "dim": (94, 240),
    "batch_size": 60,
    "n_classes": nclasses,
    "shuffle": True,
}
if model_used == 'station-generic':
    spec_paths_full = glob.glob('/Users/darrentpk/Desktop/'+ target_npy_directory + '/*.npy')
    spec_paths = []
    for spec_path in spec_paths_full:
        utc = UTCDateTime(spec_path.split('/')[-1].split('_')[1])
        spec_station = spec_path.split('/')[-1].split('_')[0]
        if starttime <= utc < endtime and spec_station in stations:
            spec_paths.append(spec_path)
    spec_placeholder_labels = [0 for i in spec_paths]
    spec_label_dict = dict(zip(spec_paths, spec_placeholder_labels))
    spec_gen = DataGenerator(spec_paths, spec_label_dict, **params)
    model_filename = '/Users/darrentpk/Desktop/GitHub/tremor_ml/models/4min_all'
    if balance_training:
        model_filename = model_filename + '_' + balance_type
    model_filename = model_filename + '_model.h5'
    saved_model = load_model(model_filename)
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
        spec_paths_full = glob.glob('/Users/darrentpk/Desktop/'+ target_npy_directory + '/*' + station + '*.npy')
        spec_paths = []
        for spec_path in spec_paths_full:
            utc = UTCDateTime(spec_path.split('/')[-1].split('_')[1])
            spec_station = spec_path.split('/')[-1].split('_')[0]
            if starttime <= utc < endtime and spec_station in stations:
                spec_paths.append(spec_path)
        spec_placeholder_labels = [0 for i in spec_paths]
        spec_label_dict = dict(zip(spec_paths, spec_placeholder_labels))
        params['batch_size'] = len(spec_paths)
        spec_gen = DataGenerator(spec_paths, spec_label_dict, **params)
        model_filename = '/Users/darrentpk/Desktop/GitHub/tremor_ml/models/4min_' + station
        if balance_training:
            model_filename = model_filename + '_' + balance_type
        model_filename = model_filename + '_model.h5'
        saved_model = load_model(model_filename)
        spec_predictions = saved_model.predict(spec_gen)
        predicted_labels = np.argmax(spec_predictions, axis=1)  # why are the lengths different?
        for i, filepath in enumerate(spec_gen.list_ids):
            filename = filepath.split('/')[-1]
            chunks = filename.split('_')
            indicators.append([chunks[0], UTCDateTime(chunks[1]), predicted_labels[i]])

# Craft unlabeled matrix
matrix_length = int((nhours*3600)/time_step)
matrix_height = nsubrows
matrix_plot = np.ones((matrix_height, matrix_length)) * nclasses
for indicator in indicators:
    if indicator[0] not in stations:
        continue
    utc = indicator[1]
    row_index = stations.index(indicator[0])
    col_index = int((indicator[1] - starttime) / time_step)
    matrix_plot[row_index, col_index] = indicator[2]

# Now add voting row
na_label = 6
new_row = np.ones((1, np.shape(matrix_plot)[1])) * na_label
matrix_plot = np.concatenate((matrix_plot,new_row))
for j in range(np.shape(matrix_plot)[1]):
    # first do it for matrix_plot
    sub_col = matrix_plot[:, j]
    labels_seen, label_counts = np.unique(sub_col, return_counts=True)
    if len(labels_seen) == 1 and na_label in labels_seen:
        new_row[0, j] = na_label
    elif len(labels_seen) == 1:
        new_row[0, j] = labels_seen[0]
    else:
        if na_label in labels_seen:
            label_counts = np.delete(label_counts, labels_seen == na_label)
            labels_seen = np.delete(labels_seen, labels_seen == na_label)
        selected_label_index = np.argwhere(label_counts == np.amax(label_counts))[-1][0]
        # selected_label_index = np.argmax(label_counts)
        new_row[0, j] = labels_seen[selected_label_index]
matrix_plot = np.concatenate((matrix_plot,new_row))

# Craft color maps
rgb_values = np.array([
    [193,  39,  45],
    [  0, 129, 118],
    [  0,   0, 167],
    [238, 204,  22],
    [164,  98,   0],
    [ 40,  40,  40],
    [255, 255, 255]])
rgb_ratios = rgb_values/255
colors = np.concatenate((rgb_ratios, np.ones((np.shape(rgb_values)[0],1))), axis=1)
cmap = ListedColormap(colors)

# Colorbar keywords
real_cbar_tick_interval = 2 * nclasses/(2*np.shape(rgb_values)[0])
real_cbar_ticks = np.arange(real_cbar_tick_interval/2, nclasses, real_cbar_tick_interval)
cbar_kws = {'ticks': real_cbar_ticks,
            'drawedges': True,
            'location': 'top',
            'fraction': 0.15,
            'aspect': 40}
            #'label': 'Classes'}

# Plot heatmap
fig, ax1 = plt.subplots(figsize=(16, 2.5))
sns.heatmap(matrix_plot, cmap=cmap, cbar=False, cbar_kws=cbar_kws, alpha=0.8, vmin=0, vmax=nclasses, ax=ax1)
# for i in range(1,nhours):
#     ax1.axvline(x=(i * (3600 / time_step)), linestyle='--', linewidth=3, color='k')
ax1.set_xticks([])
# ax1.set_xticks(np.arange(0, matrix_length, 3600 / (time_step)))
# ax1.set_xticklabels([(starttime + 3600 * i).strftime('%y/%m/%d %H:%M') for i in range(nhours)], rotation=0)
ax1.axhline(nsubrows, color='black')
ax1.axhline(nsubrows+1, color='black')
ax1.axhspan(nsubrows, nsubrows+1, facecolor='lightgray')
yticks = np.concatenate((np.arange(0.5,nsubrows,1),np.array([nsubrows+1.5])))
ax1.set_yticks(yticks)
yticklabels = stations.copy()
yticklabels.append('VOTE')
ax1.set_yticklabels(yticklabels, rotation=0, fontsize=20)
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth(2)
ax1.set_title('Station-based Voting', fontsize=24)
plt.show()

# Plot spectrogram
if plot_spectrogram:
    source = 'IRIS'
    network = 'AV'
    station = ','.join(stations)
    location = ''
    channel = '*HZ'
    pad = 60  # padding length [s]
    local = False  # pull data from local
    data_dir = None  # local data directory if pulling data from local
    client = 'IRIS'  # FDSN client for data pull
    filter_band = None  # frequency band for bandpass filter
    window_duration = 10  # spectrogram window duration [s]
    freq_lims = (0.5, 10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
    log = False  # logarithmic scale in spectrogram
    demean = False  # remove the temporal mean of the plotted time span from the spectrogram matrix
    v_percent_lims = (20, 97.5)  # colorbar limits
    export_path = None
    export_spec = False  # export spectrogram with no axis labels
    verbose = False
    # Load data using waveform_collection tool
    stream = gather_waveforms(source=source, network=network, station=station, location=location, channel=channel,
                              starttime=starttime - pad, endtime=endtime + pad, verbose=False)
    # Process waveform
    stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=pad, taper_percentage=None,
                              filter_band=filter_band, verbose=False)
    # Plot all spectrograms on one figure
    plot_spectrogram_multi(stream, starttime, endtime, window_duration, freq_lims, log=log, demean=demean,
                           v_percent_lims=v_percent_lims, earthquake_times=None, db_hist=False,
                           export_path=export_path, export_spec=export_spec)

if plot_metrics:
    # Load data using waveform_collection tool
    source = 'IRIS'
    network = 'AV'
    station = ','.join(stations)
    location = ''
    channel = '*HZ'
    pad = 360  # padding length [s]
    stream = gather_waveforms(source=source, network=network, station=station, location=location, channel=channel,
                              starttime=starttime - pad, endtime=endtime + pad, verbose=False)
    # Compute metrics
    tmpl, rsam, dr, pe, fc, fd, fsd = compute_metrics(stream, process_taper=60, metric_taper=pad, filter_band=(0.5,10),
                                                      window_length=time_step, overlap=0.5)

    # Prepare time ticks
    if ((endtime-starttime)%1800) == 0:
        denominator = 12
        time_tick_list = np.arange(starttime,endtime+1,(endtime-starttime)/denominator)
    else:
        denominator = 10
        time_tick_list = np.arange(starttime,endtime+1,(endtime-starttime)/denominator)
    time_tick_list_mpl = [t.matplotlib_date for t in time_tick_list]
    time_tick_labels = [time.strftime('%H:%M') for time in time_tick_list]

    # # Craft figure:
    # fig, axs = plt.subplots(6, 1, figsize=(32, 24))
    # fig.subplots_adjust(hspace=0)
    # for i in range(np.shape(tmpl)[0]):
    #     axs[0].plot(tmpl[i,:], rsam[i,:])
    #     axs[0].set_ylabel('RSAM ($\mu$m)', fontsize=22, fontweight='bold')
    #     axs[1].plot(tmpl[i,:], dr[i,:])
    #     axs[1].set_ylabel('D_R', fontsize=22, fontweight='bold')
    #     axs[2].plot(tmpl[i,:], pe[i,:])
    #     axs[2].set_ylabel('p_e', fontsize=22, fontweight='bold')
    #     axs[3].plot(tmpl[i,:], fc[i,:])
    #     axs[3].set_ylabel('f_c', fontsize=22, fontweight='bold')
    #     axs[4].plot(tmpl[i,:], fd[i,:])
    #     axs[4].set_ylabel('f_d', fontsize=22, fontweight='bold')
    #     axs[5].plot(tmpl[i,:], fsd[i,:],label=str(i))
    #     axs[5].set_ylabel('f_sd', fontsize=22, fontweight='bold')
    # for j in range(6):
    #     axs[j].set_xlim([starttime.matplotlib_date, endtime.matplotlib_date])
    #     axs[j].tick_params(axis='y', labelsize=18)
    # axs[-1].legend(fontsize=20)
    # axs[-1].set_xticks(time_tick_list_mpl)
    # axs[-1].set_xticklabels(time_tick_labels, fontsize=22, rotation=30)
    # axs[-1].set_xlabel('UTC Time on ' + starttime.date.strftime('%b %d, %Y'), fontsize=25)
    # fig.show()

    fig, axs = plt.subplots(6, 1, figsize=(32, 24))
    fig.subplots_adjust(hspace=0)
    axs[0].plot(tmpl[0,:], np.median(rsam,axis=0))
    axs[0].set_ylabel('RSAM ($\mu$m)', fontsize=22, fontweight='bold')
    axs[1].plot(tmpl[0,:], np.median(dr,axis=0))
    axs[1].set_ylabel('D_R', fontsize=22, fontweight='bold')
    axs[2].plot(tmpl[0,:], np.median(pe,axis=0))
    axs[2].set_ylabel('p_e', fontsize=22, fontweight='bold')
    axs[3].plot(tmpl[0,:], np.median(fc,axis=0))
    axs[3].set_ylabel('f_c', fontsize=22, fontweight='bold')
    axs[4].plot(tmpl[0,:], np.median(fd,axis=0))
    axs[4].set_ylabel('f_d', fontsize=22, fontweight='bold')
    axs[5].plot(tmpl[0,:], np.median(fsd,axis=0),label=str(i))
    axs[5].set_ylabel('f_sd', fontsize=22, fontweight='bold')
    for j in range(6):
        axs[j].set_xlim([starttime.matplotlib_date, endtime.matplotlib_date])
        axs[j].tick_params(axis='y', labelsize=18)
    axs[-1].set_xticks(time_tick_list_mpl)
    axs[-1].set_xticklabels(time_tick_labels, fontsize=22, rotation=30)
    axs[-1].set_xlabel('UTC Time on ' + starttime.date.strftime('%b %d, %Y'), fontsize=25)
    fig.show()
