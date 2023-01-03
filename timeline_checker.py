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
from toolbox import process_waveform, plot_spectrogram_multi

# Define variables
model_used = 'station-specific'  # or 'station-generic'
stations = ['PN7A', 'PS1A', 'PS4A', 'PV6A', 'PVV']  # ORDER MATTERS
nsubrows = len(stations)
nclasses = 6
time_step = 4 * 60  # s
nhours = 4
T1 = UTCDateTime(2022, 7, 21, 0, 0, 0)
T2 = UTCDateTime(2022, 7, 22, 0, 0, 0)
for starttime in np.arange(T1,T2,nhours*3600):
    # starttime = UTCDateTime(2021,9,1,0,0,0)
    endtime = starttime + nhours*3600
    comparison = True

    # Make predictions with model
    params = {
        "dim": (94, 240),
        "batch_size": 60,
        "n_classes": nclasses,
        "shuffle": True,
    }
    if model_used == 'station-generic':
        spec_paths_full = glob.glob('/Users/darrentpk/Desktop/all_npy/*.npy')
        spec_paths = []
        for spec_path in spec_paths_full:
            utc = UTCDateTime(spec_path.split('/')[-1].split('_')[1])
            if starttime <= utc < endtime:
                spec_paths.append(spec_path)
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
            spec_paths_full = glob.glob('/Users/darrentpk/Desktop/all_npy/*' + station + '*.npy')
            spec_paths = []
            for spec_path in spec_paths_full:
                utc = UTCDateTime(spec_path.split('/')[-1].split('_')[1])
                if starttime <= utc < endtime:
                    spec_paths.append(spec_path)
            spec_placeholder_labels = [0 for i in spec_paths]
            spec_label_dict = dict(zip(spec_paths, spec_placeholder_labels))
            params['batch_size'] = len(spec_paths)
            spec_gen = DataGenerator(spec_paths, spec_label_dict, **params)
            saved_model = load_model('/Users/darrentpk/Desktop/GitHub/tremor_ml/models/4min_' + station +'_model.h5')
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
        utc = indicator[1]
        row_index = stations.index(indicator[0])
        col_index = int((indicator[1] - starttime) / time_step)
        matrix_plot[row_index, col_index] = indicator[2]

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

    # Colorbar keywords
    real_cbar_tick_interval = 2 * nclasses/(2*np.shape(rgb_values)[0])
    real_cbar_ticks = np.arange(real_cbar_tick_interval/2, nclasses, real_cbar_tick_interval)
    cbar_kws = {'ticks': real_cbar_ticks,
                'drawedges': True,
                'location': 'top',
                'fraction': 0.15,
                'aspect': 40}
                #'label': 'Classes'}

    # Craft timeline figure
    if comparison:
        # Plot heatmap
        fig, ax1 = plt.subplots(figsize=(20, 2))
        sns.heatmap(matrix_plot, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.7, vmin=0, vmax=nclasses, ax=ax1)
        cbar = ax1.collections[0].colorbar
        cbar.outline.set_color('black')
        cbar.outline.set_linewidth(1.5)
        cbar.ax.set_xticklabels(['Broadband',
                                 'Harmonic',
                                 'Monochromatic',
                                 'Non-Tremor',
                                 'Explosion',
                                 'Noise',
                                 'N/A'], fontsize=12)
        for i in range(nhours):
            ax1.axvline(x=(i * (3600 / time_step)), linestyle='--', linewidth=3, color='k')
        ax1.set_xticks(np.arange(0, matrix_length, 3600 / (time_step)))
        ax1.set_xticklabels([(starttime + 3600 * i).strftime('%y/%m/%d %H:%M') for i in range(nhours)], rotation=0)
        ax1.set_yticks(np.arange(0.5, len(stations), 1))
        ax1.set_yticklabels(stations, rotation=0, fontsize=14)
        ax1.patch.set_edgecolor('black')
        ax1.patch.set_linewidth(2)
        plt.show()
        # Plot spectrogram
        source = 'IRIS'
        network = 'AV'
        station = 'PN7A,PS1A,PS4A,PV6A,PVV'
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
                               v_percent_lims=v_percent_lims, earthquake_times=None, db_hist=True,
                               export_path=export_path, export_spec=export_spec)
    else:
        fig, ax1 = plt.subplots(figsize=(20, 2))
        sns.heatmap(matrix_plot, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.7, ax=ax1)
        cbar = ax1.collections[0].colorbar
        cbar.outline.set_color('black')
        cbar.outline.set_linewidth(1.5)
        cbar.ax.set_xticklabels(['Broadband',
                                 'Harmonic',
                                 'Monochromatic',
                                 'Non-Tremor',
                                 'Explosion',
                                 'Noise',
                                 'N/A'], fontsize=12)
        for i in range(nhours):
            ax1.axvline(x=(i * (3600 / time_step)), linestyle='--', linewidth=3, color='k')
        ax1.set_xticks(np.arange(0, matrix_length, 3600 / (time_step)))
        ax1.set_xticklabels([(starttime + 3600 * i).strftime('%y/%m/%d %H:%M') for i in range(nhours)], rotation=0)
        ax1.set_yticks(np.arange(0.5, len(stations), 1))
        ax1.set_yticklabels(stations, rotation=0, fontsize=14)
        ax1.patch.set_edgecolor('black')
        ax1.patch.set_linewidth(2)
        plt.show()