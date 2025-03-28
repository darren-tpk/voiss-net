# Import dependencies
import os
import time
import glob
import json
import pickle
import random
import pyproj
import numpy as np
import pandas as pd
import colorcet as cc
import seaborn as sns
import tensorflow as tf
import statistics as sts
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator
from geopy.distance import geodesic as GD
from keras import layers, models, losses, optimizers
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from matplotlib import dates, rcParams
from matplotlib.transforms import Bbox
from matplotlib.colors import ListedColormap
from obspy import UTCDateTime, read, Stream, Trace
from ordpy import complexity_entropy
from scipy.signal import spectrogram, find_peaks, medfilt
from scipy.fft import rfft, rfftfreq
from sklearn import metrics
from waveform_collection import gather_waveforms
from toolbox import process_waveforms, calculate_spectrogram

def create_labeled_dataset(json_filepath, output_dir, label_dict, transient_indices, time_step, source, network, station, location, channel, pad, window_duration, freq_lims, transient_ratio=0.1):
    """
    Create a labeled spectrogram dataset from a json file from label studio
    :param json_filepath (str): File path to the json file from label studio
    :param output_dir (str): Directory to save the npy files
    :param label_dict (dict): Dictionary to convert labels to appended file index
    :param transient_indices (list): List of indices for transients (these will be prioritized in assigning labels, and the higher index number will be prioritized)
    :param time_step (float): Time step for the 2D matrices
    :param source (str): Source of the data
    :param network (str or list): SEED network code(s)
    :param station (str or list): SEED station code(s)
    :param channel (str or list): SEED channel code(s)
    :param location (str or list): SEED location code(s)
    :param pad (float): Padding length [s]
    :param window_duration (float): Window duration for the spectrogram [s]
    :param freq_lims (tuple): Tuple of length 2 storing minimum frequency and maximum frequency for the spectrogram plot ([Hz],[Hz])
    :param transient_ratio (float): Ratio of transient-related time samples for transient classes to be prioritized (default: 0.1)
    """

    # Check if output directory exists
    if not os.path.exists(output_dir):
        raise ValueError('Output directory does not exist')

    # Parse json file from label studio
    f = open(json_filepath)
    labeled_images = json.load(f)
    f.close()

    # Loop over all labeled images:
    for labeled_image in labeled_images:

        # Extract out file name, and define starttime, endtime and stations covered by spectrogram image
        filename = labeled_image['file_upload'].split('-')[1]
        chunks = filename.split('_')
        t1 = UTCDateTime(chunks[0] + chunks[1])
        t2 = UTCDateTime(chunks[3] + chunks[4])
        stations = chunks[5:-1]

        # Extract all annotations
        annotations = labeled_image['annotations'][0]['result']

        # If no annotations exist, skip
        if len(annotations) == 0:
            print('No annotations on image')
            continue
        # Otherwise define original width and height of image in pixels and determine pixels indicating each station
        else:
            time_per_percent = (t2 - t1) / 100
            y_span = annotations[0]['original_height']
            y_per_percent = y_span / 100
            station_indicators = np.arange(y_span / (len(stations) * 2), y_span, y_span / (len(stations)))

        # Initialize time bound list
        time_bounds = []

        # Now loop over annotations to fill
        for annotation in annotations:
            if annotation['value']['rectanglelabels'] == []:
                continue
            label = annotation['value']['rectanglelabels'][0]
            x1 = t1 + (annotation['value']['x'] * time_per_percent)
            x2 = t1 + ((annotation['value']['x'] + annotation['value']['width']) * time_per_percent)
            y1 = (annotation['value']['y'] * y_per_percent)
            y2 = ((annotation['value']['y'] + annotation['value']['height']) * y_per_percent)
            stations_observed = [stations[i] for i in range(len(stations))
                                 if (station_indicators[i] > y1 and station_indicators[i] < y2)]
            for station_observed in stations_observed:
                time_bound = [station_observed, x1, x2, label]
                time_bounds.append(time_bound)

        # Load data using waveform_collection tool
        successfully_loaded = False
        while not successfully_loaded:
            try:
                # Gather waveforms
                stream = gather_waveforms(source=source, network=network, station=station, location=location,
                                          channel=channel, starttime=t1 - pad, endtime=t2 + pad, verbose=False)

                # Process waveform
                stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=pad,
                                          taper_percentage=None, filter_band=None, verbose=False)
                successfully_loaded = True
            except:
                print('gather_waveforms failed to retrieve response')
                pass

        # Loop over stations that have data
        stream_stations = [tr.stats.station for tr in stream]
        for i, stream_station in enumerate(stream_stations):

            # Choose trace corresponding to station
            trace = stream[i]

            # Calculate spectrogram power matrix
            spec_db, utc_times = calculate_spectrogram(trace, t1, t2, window_duration, freq_lims, demean=False)

            # Get label time bounds that are observed on station
            time_bounds_station = [tb for tb in time_bounds if tb[0] == stream_station]

            # Define array of time steps for spectrogram slicing
            step_bounds = np.arange(t1, t2 + time_step, time_step)

            # Loop over time steps
            for j in range(len(step_bounds) - 1):

                # Slice spectrogram
                sb1 = step_bounds[j]
                sb2 = step_bounds[j + 1]
                spec_slice_indices = np.flatnonzero([sb1 < t < sb2 for t in utc_times])
                spec_slice = spec_db[:, spec_slice_indices]

                # Enforce spectrogram length
                if np.shape(spec_slice)[1] != time_step:
                    # Try inclusive slicing time span (<= sb2)
                    spec_slice_indices = np.flatnonzero([sb1 < t <= sb2 for t in utc_times])
                    spec_slice = spec_db[:, spec_slice_indices]
                    # If it still doesn't fit our shape, try double inclusive time bounds
                    if np.shape(spec_slice)[1] != time_step:
                        spec_slice_indices = np.flatnonzero([sb1 <= t <= sb2 for t in utc_times])
                        spec_slice = spec_db[:, spec_slice_indices]
                        # Otherwise, raise error
                        if np.shape(spec_slice)[1] != time_step:
                            raise ValueError('THE SHAPE IS NOT RIGHT.')

                # Skip matrices that have a spectrogram data gap
                amplitude_threshold = 0 if channel[-2:] == 'DF' else -220
                if np.sum(spec_slice.flatten() < amplitude_threshold) > (0.2 * time_step):
                    print('Skipping due to data gap, %d elements failed the check' % np.sum(spec_slice.flatten() < amplitude_threshold))
                    continue

                # Obtain corresponding time samples for spectrogram slice
                time_slice = utc_times[spec_slice_indices]

                # Check for overlaps and fill a vector with labels to decide final label
                label_indices = np.ones(len(time_slice)) * -1
                for time_bound_station in time_bounds_station:

                    # Labeled time bound starts in slice and ends in slice
                    if time_bound_station[1] >= sb1 and time_bound_station[2] <= sb2:
                        valid_indices = np.flatnonzero(
                            np.logical_and(time_slice >= time_bound_station[1], time_slice <= time_bound_station[2]))
                        label_indices[valid_indices] = label_dict[time_bound_station[3]]

                    # Labeled time bound starts before slice and ends after slice
                    elif time_bound_station[1] < sb1 and time_bound_station[2] > sb2:
                        label_indices[:] = label_dict[time_bound_station[3]]

                    # Labeled time bound starts before slice and ends in slice
                    elif time_bound_station[1] < sb1 and (sb1 <= time_bound_station[2] <= sb2):
                        label_indices[np.flatnonzero(time_slice <= time_bound_station[2])] = label_dict[
                            time_bound_station[3]]

                    # Labeled time bound starts in slice and ends after slice
                    elif (sb1 <= time_bound_station[1] <= sb2) and time_bound_station[2] > sb2:
                        label_indices[np.flatnonzero(time_slice >= time_bound_station[1])] = label_dict[
                            time_bound_station[3]]

                # Count how many time samples correspond to each label
                labels_seen, label_counts = np.unique(label_indices, return_counts=True)

                # Define dummy label
                final_label = -1

                # Check for tremor or noise label in > 50 % of the time samples
                if np.max(label_counts) > 0.5 * len(label_indices) and sts.mode(label_indices) != -1:
                    final_label_value = int(labels_seen[np.argmax(label_counts)])
                    final_label = next(key for key, value in label_dict.items() if value == final_label_value)

                # Override label with transient label if it is in > 10 % of the time samples
                if len(set(labels_seen) & set(transient_indices)) != 0:
                    for transient_index in list(set(labels_seen) & set(transient_indices)):
                        if label_counts[list(labels_seen).index(transient_index)] >= transient_ratio * len(label_indices):
                            final_label = list(label_dict.keys())[int(transient_index)]

                # If label is still invalid, skip
                if final_label == -1:
                    continue

                # Construct file name and save
                file_name = stream_station + '_' + sb1.strftime('%Y%m%d%H%M') + '_' + \
                            sb2.strftime('%Y%m%d%H%M') + '_' + str(label_dict[final_label]) + '.npy'
                np.save(output_dir + file_name, spec_slice)

    print('Done.')

def augment_labeled_dataset(npy_dir,omit_index,noise_index,testval_ratio,noise_ratio,plot_example=False):
    """
    Use noise-adding augmentation strategy to generate lists of balanced train, validation and testfile paths.
    :param npy_dir (str): directory to retrieve raw labeled files and create nested augmented file directory
    :param omit_index (list of int): class index to omit when calculating augmented number (extras will be discarded)
    :param noise_index (int): class index pointing to the noise class. Files with this class index will be randomly sampled for augmentation.
    :param testval_ratio (float): ratio of file counts set aside for the test set and validation set, each (note this will be calculated on the sparsest class count)
    :param noise_ratio (float): ratio of noise used to generated augmented samples [augmented_image = (1-noise_ratio) * augment_image) + (noise_ratio * noise_image)]
    :param plot_example (bool): if set to `True`, generate a plot showing examples of augmented images
    :return: list: list of training set filepaths (combining both raw and augmented filepaths)
    :return: list: list of validation set filepaths
    :return: list: list of test set filepaths
    """

    # Count the number of samples of each class
    nclasses = len(np.unique([filepath[-5] for filepath in glob.glob(npy_dir + '*.npy')]))
    class_paths = [glob.glob(npy_dir + '*_' + str(c) + '.npy') for c in range(nclasses)]
    class_counts = np.array([len(paths) for paths in class_paths])
    print('\nInitial class counts:')
    print(''.join(['%d: %d\n' % (c,class_counts[c]) for c in range(nclasses)]))

    # Determine number of samples set aside for validation set and test set (each)
    testval_number = int(np.floor(np.min(class_counts)/(1/testval_ratio)))
    print('Setting aside samples for validation and test set (%.1f%% of sparsest class count each)' % (testval_ratio*100))
    print('%d samples kept for validation set (%d per class)' % (nclasses*testval_number,testval_number))
    print('%d samples kept for test set (%d per class)' % (nclasses*testval_number,testval_number))

    # Calculate augmented number
    print('\nCalculating augmented number...')
    leftover_counts = class_counts - 2*testval_number
    augmented_number = np.mean(leftover_counts[[i for i in range(nclasses) if i not in omit_index]])
    augmented_number = int(np.min([augmented_number] + list(leftover_counts[omit_index])))
    print('Class index %s are omitted and class index %d will be used as noise samples...' % (str(','.join([str(i) for i in omit_index])),noise_index))
    print('%d samples will be gathered for training set (%d per class)' % (nclasses*augmented_number,augmented_number))

    # Determine test and validation sample list
    test_list = []
    val_list = []
    keep_list = []
    for c in range(nclasses):
        test_list = test_list + list(np.random.choice(class_paths[c], testval_number, replace=False))
        leftover_list = [filepath for filepath in class_paths[c] if filepath not in test_list]
        val_list = val_list + list(np.random.choice(leftover_list, testval_number, replace=False))
        leftover_list = [filepath for filepath in leftover_list if filepath not in val_list]
        if c in omit_index:
            keep_list = keep_list + list(np.random.choice(leftover_list, augmented_number, replace=False))
        elif c == noise_index:
            keep_list = keep_list + list(np.random.choice(leftover_list, augmented_number, replace=False))
            noise_list = [filepath for filepath in leftover_list if filepath not in keep_list]
        elif len(leftover_list) <= augmented_number:
            keep_list = keep_list + leftover_list
        else:
            raise ValueError('Class index %d has more samples than the augment number. Check class counts!' % c)

    # Commence augmentation
    print('\nCreating nested augmented directory and commencing augmentation...')

    # Create a temporary directory if it does not exist
    if not os.path.exists(npy_dir + 'augmented/'):
        os.mkdir(npy_dir + 'augmented/')

    # Clear all existing files in augmented subfolder if any
    for f in glob.glob(npy_dir + 'augmented/*.npy'):
        os.remove(f)

    # Randomly sample based on count difference
    aug_list = []
    for c in range(nclasses):
        if c in omit_index or c == noise_index:
            continue
        else:
            keep_sublist = [f for f in keep_list if int(f.split('_')[-1][0]) == c]
            count_difference = augmented_number - len(keep_sublist)
            aug_list = aug_list + list(np.random.choice(keep_sublist, count_difference, replace=True))

    # Check if augment list and noise list have the same length
    if len(aug_list) == len(noise_list):
        print('Augmentation list and noise list match in length. Proceeding...\n')
    else:
        print('Augmentation list and noise list do NOT match in length. Noise list will be trimmed.')
        noise_list = list(np.random.choice(noise_list, len(aug_list), replace=False))

    # Shuffle and add noise to augment samples
    print('Shuffling and adding noise samples to augment samples...')
    random.shuffle(aug_list)
    random.shuffle(noise_list)
    for augment_sample, noise_sample in zip(aug_list,noise_list):

        # Load both images and sum them using noise ratio
        augment_image = np.load(augment_sample)
        noise_image = np.load(noise_sample)
        augmented_image = ((1-noise_ratio) * augment_image) + (noise_ratio * noise_image)

        # Determine filepath by checking for uniqueness
        n = 0
        augmented_filepath = npy_dir + 'augmented/' + augment_sample.split('/')[-1][:-4] + 'aug' + str(n) + '.npy'
        while os.path.isfile(augmented_filepath):
            n += 1
            augmented_filepath = npy_dir + 'augmented/' + augment_sample.split('/')[-1][:-4] + 'aug' + str(n) + '.npy'
        # Save augmented image as a unique file
        np.save(augmented_filepath, augmented_image)

    # Compile train list
    train_list = glob.glob(npy_dir + 'augmented/*.npy') + keep_list
    print('Done!')

    # Plot examples if desired
    if plot_example:
        import matplotlib.pyplot as plt
        import colorcet as cc
        indices = np.random.choice(range(len(aug_list)), 5)
        fig, ax = plt.subplots(5, 3, figsize=(4.2, 10))
        for i, n in enumerate(indices):
            augment_image = np.load(aug_list[n])
            noise_image = np.load(noise_list[n])
            augmented_image = ((1 - noise_ratio) * augment_image) + (noise_ratio * noise_image)
            ax[i,0].imshow(augment_image, vmin=np.percentile(augment_image, 20), vmax=np.percentile(augment_image, 97.5),
                         origin='lower', aspect='auto', interpolation='None', cmap=cc.cm.rainbow)
            ax[i,0].set_xticks([])
            ax[i,0].set_yticks([])
            ax[i,0].set_title('Class ' + str(aug_list[n].split('_')[-1][0]) + '', fontsize=10)
            ax[i,1].imshow(noise_image, vmin=np.percentile(noise_image, 20), vmax=np.percentile(noise_image, 97.5),
                         origin='lower', aspect='auto', interpolation='None', cmap=cc.cm.rainbow)
            ax[i,1].set_xticks([])
            ax[i,1].set_yticks([])
            ax[i,1].set_title('Noise ' + str('%.2f' % noise_ratio), fontsize=10)
            ax[i,2].imshow(augmented_image, vmin=np.percentile(augmented_image, 20),
                         vmax=np.percentile(augmented_image, 97.5),
                         origin='lower', aspect='auto', interpolation='None', cmap=cc.cm.rainbow)
            ax[i,2].set_xticks([])
            ax[i,2].set_yticks([])
            ax[i,2].set_title('Augmented ', fontsize=10)
        fig.show()

    return train_list, val_list, test_list

def train_voiss_net(train_paths, valid_paths, test_paths, label_dict, model_tag, batch_size=100, learning_rate=0.0005, patience=20, meanvar_standardization=False):
    """
    Train an iteration VOISS-Net convolutional neural network for classifying spectrogram slices
    :param train_paths (list): List of file paths to training data
    :param valid_paths (list): List of file paths to validation data
    :param test_paths (list): List of file paths to test data
    :param label_dict (dict): Dictionary to convert appended file index to their actual label
    :param model_tag (str): Tag for model and plot file names (e.g., model will be saved as './models/[model_tag]_model.keras')
    :param batch_size (int): Batch size for training (default 100)
    :param learning_rate (float): Learning rate for the optimizer (default 0.0005)
    :param patience (int): Number of epochs to  for early stopping (default 20)
    :param meanvar_standardization (bool): Whether to standardize spectrograms by mean and variance (default `False`). If `True`, the mean and variance of the training data will be used to standardize the validation and test data, and a output mean and variacne will be saved.
    """

    # Initialize model and figure subdirectories if needed
    if not os.path.isdir('./models/'):
        os.mkdir('./models/')
    if not os.path.isdir('./figures/'):
        os.mkdir('./figures/')

    # Define filepaths of all model products
    model_filepath = './models/' + model_tag + '_model.keras'
    history_filepath = './models/' + model_tag + '_history.npy'
    curve_filepath = './figures/' + model_tag + '_curve.png'
    confusion_filepath = './figures/' + model_tag + '_confusion.png'
    predictions_filepath = './models/' + model_tag + '_predictions.npy'
    if meanvar_standardization:
        meanvar_filepath = './models/' + model_tag + '_meanvar.npy'

    # Determine the number of unique classes in training to decide on the number of output nodes in model
    train_classes = [int(i.split("_")[-1][0]) for i in train_paths]
    unique_classes = np.unique(train_classes)

    # Read in example file to determine spectrogram shape
    eg_spec = np.load(train_paths[0])

    # Define parameters to generate the training, testing, validation data
    if meanvar_standardization:
        params = {
            "dim": eg_spec.shape,
            "n_classes": len(unique_classes),
            "shuffle": True,
            "running_x_mean": np.mean(eg_spec),
            "running_x_var": np.var(eg_spec)
        }
    else:
        params = {
            "dim": eg_spec.shape,
            "n_classes": len(unique_classes),
            "shuffle": True,
            "running_x_mean": None,
            "running_x_var": None}

    # Configure test and validation lists
    valid_classes = [int(i.split("_")[-1][0]) for i in valid_paths]
    test_classes = [int(i.split("_")[-1][0]) for i in test_paths]

    # Create a dictionary holding labels for all filepaths in the training, testing, and validation data
    train_labels = [np.where(i == unique_classes)[0][0] for i in train_classes]
    train_label_dict = dict(zip(train_paths, train_labels))
    valid_labels = [np.where(i == unique_classes)[0][0] for i in valid_classes]
    valid_label_dict = dict(zip(valid_paths, valid_labels))
    test_labels = [np.where(i == unique_classes)[0][0] for i in test_classes]
    test_label_dict = dict(zip(test_paths, test_labels))

    # Initialize the training and validation generators
    train_gen = DataGenerator(train_paths, train_label_dict, batch_size=batch_size, **params, is_training=True)
    valid_gen = DataGenerator(valid_paths, valid_label_dict, batch_size=len(valid_paths), **params, is_training=False)

    # Define a Callback class that allows the validation dataset to adopt the running
    # training mean and variance for spectrogram standardization (by each pixel)
    if meanvar_standardization:
        class ExtractMeanVar(Callback):
           def on_epoch_end(self, epoch, logs=None):
               valid_gen.running_x_mean = train_gen.running_x_mean
               valid_gen.running_x_var = train_gen.running_x_var

    # Define the CNN

    # Add singular channel to conform with tensorflow input dimensions
    input_shape = [*eg_spec.shape, 1]

    # Build a sequential model
    model = models.Sequential()
    # Convolutional layer, 32 filters, 3x3 kernel, 1x1 stride, padded to retain shape
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape, padding="same"))
    # Max pooling layer, 3x3 pool, 3x3 stride
    model.add(layers.MaxPooling2D((3, 3)))
    # Convolutional layer, 64 filters, 3x3 kernel, 1x1 stride, padded to retain shape
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    # Max pooling layer, 3x3 pool, 3x3 stride
    model.add(layers.MaxPooling2D((3, 3)))
    # Convolutional layer, 128 filters, 3x3 kernel, 1x1 stride, padded to retain shape
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    # Max pooling layer, 3x3 pool, 3x3 stride
    model.add(layers.MaxPooling2D((3, 3)))

    # Flatten and add 20% dropout to inputs
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))

    # Dense layer, 128 units with 50% dropout
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))

    # Dense layer, 64 units with 50% dropout
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))
    # Dense layer, 6 units, one per class
    model.add(layers.Dense(params["n_classes"], activation="softmax"))
    # Compile model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=["accuracy"])
    # Print out model summary
    model.summary()
    # Implement early stopping, checkpointing, and transference of mean and variance
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint(model_filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # Fit model
    if not meanvar_standardization:
        history = model.fit(train_gen, validation_data=valid_gen, epochs=200, callbacks=[es, mc])
    else:
        emv = ExtractMeanVar()
        history = model.fit(train_gen, validation_data=valid_gen, epochs=200, callbacks=[es, mc, emv])
        np.save(meanvar_filepath, [train_gen.running_x_mean, train_gen.running_x_var])

    # Save model training history to reproduce learning curves
    np.save(history_filepath, history.history)

    # Plot loss and accuracy curves
    fig_curves, axs = plt.subplots(1, 2, figsize=(8, 5))
    axs[0].plot(history.history["accuracy"], label="Training", lw=1)
    axs[0].plot(history.history["val_accuracy"], label="Validation", lw=1)
    axs[0].axvline(len(history.history["val_accuracy"]) - es.patience - 1, color='k', linestyle='--', alpha=0.5,
                   label='Early Stop')
    axs[0].set_ylim([0, 1])
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[1].plot(history.history["loss"], label="Training", lw=1)
    axs[1].plot(history.history["val_loss"], label="Validation", lw=1)
    axs[1].axvline(len(history.history["val_loss"]) - es.patience - 1, color='k', linestyle='--', alpha=0.5,
                   label='Early Stop')
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[0].legend()
    fig_curves.suptitle(model_tag, fontweight='bold')
    fig_curves.savefig(curve_filepath, bbox_inches='tight')
    fig_curves.show()

    # Create data generator for test data
    test_params = {
        "dim": eg_spec.shape,
        "batch_size": len(test_labels),
        "n_classes": len(unique_classes),
        "shuffle": False}
    if not meanvar_standardization:
        test_gen = DataGenerator(test_paths, test_label_dict, **test_params, is_training=False,
                                 running_x_mean=None, running_x_var=None)
    else:
        test_gen = DataGenerator(test_paths, test_label_dict, **test_params, is_training=False,
                                 running_x_mean=train_gen.running_x_mean, running_x_var=train_gen.running_x_var)

    # Use saved model to make predictions
    saved_model = load_model(model_filepath)
    test = saved_model.predict(test_gen)
    pred_labs = np.argmax(test, axis=1)
    true_labs = np.array([test_gen.labels[id] for id in test_gen.list_ids])

    # Save predictions
    np.save(predictions_filepath, (true_labs, pred_labs))

    # Print evaluation on test data
    acc = metrics.accuracy_score(true_labs, pred_labs)
    pre, rec, f1, _ = metrics.precision_recall_fscore_support(true_labs, pred_labs, average='macro')
    metrics_chunk = model_filepath + '\n' + ('Accuracy: %.3f' % acc) + '\n' + ('Precision: %.3f' % pre) + '\n' + (
                'Recall: %.3f' % rec) + '\n' + ('F1 Score: %.3f' % f1)
    print(metrics_chunk)

    # Confusion matrix
    class_labels_raw = list(label_dict.keys())
    class_labels = [cl.replace(' ', '\n') for cl in class_labels_raw]
    confusion_matrix = metrics.confusion_matrix(true_labs, pred_labs, normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=class_labels)
    cm_display.plot(xticks_rotation=30, cmap=cc.cm.blues, values_format='.2f', im_kw=dict(vmin=0, vmax=1))
    fig_cm = plt.gcf()
    fig_cm.set_size_inches(8, 6.25)
    plt.title(model_tag + '\nAccuracy: %.3f, Precision :%.3f,\nRecall:%.3f, F1 Score:%.3f' % (acc, pre, rec, f1),
              fontweight='bold')
    fig_cm.savefig(confusion_filepath, bbox_inches='tight')
    fig_cm.show()

    # Print text on completion
    print('Done.')


def set_universal_seed(seed_value):
    """
    Reset seed for all applicable randomizers
    :param seed_value  (int): desired randomization seed number
    :return: None
    """

    # Import dependencies
    import os
    import random
    import numpy as np
    import tensorflow as tf
    #from keras import backend as K
    from tensorflow.python.keras import backend as K

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #tf.compat.v1.keras.backend.set_session(sess)
    K.set_session(sess)

def split_labeled_dataset(npy_dir,testval_ratio,stratified,max_train_samples=None):
    """
    Split labeled filepaths using the given test/validation set ratio by class proportions
    :param npy_dir (str): directory to retrieve raw labeled files
    :param testval_ratio (float): ratio of file counts set aside for the test set and validation set, each
    :param stratified (bool): if `True`, testval_ratio will be applied to each class independently, otherwise the sparse-est class will be the reference
    :param max_train_samples (int): if not `None`, impose a maximum sample count for the training set per class (to toss out excessive samples))
    :return: list: list of training set filepaths
    :return: list: list of validation set filepaths
    :return: list: list of test set filepaths
    """

    # Count the number of samples of each class
    nclasses = len(np.unique([filepath[-5] for filepath in glob.glob(npy_dir + '*.npy')]))
    class_paths = [glob.glob(npy_dir + '*_' + str(c) + '.npy') for c in range(nclasses)]
    class_counts = np.array([len(paths) for paths in class_paths])
    print('\nInitial class counts:')
    print(''.join(['%d: %d\n' % (c, class_counts[c]) for c in range(nclasses)]))

    # Determine number of samples set aside for validation set and test set (each)
    print('Train-val-test split (%.1f-%.1f-%.1f) with stratified=%s and max_train_samples=%s:' % ((1-testval_ratio*2)*100,testval_ratio*100,testval_ratio*100,stratified,str(max_train_samples)))
    if stratified:
        testval_numbers = [int(n) for n in (np.floor(class_counts * testval_ratio))]
    else:
        testval_number = int(np.floor(np.min(class_counts)/(1/testval_ratio)))

    # Return random sampled list
    train_list = []
    val_list = []
    test_list = []
    for c in range(nclasses):
        if stratified:
            testval_number = testval_numbers[c]
        test_list = test_list + list(np.random.choice(class_paths[c], testval_number, replace=False))
        leftover_list = [filepath for filepath in class_paths[c] if filepath not in test_list]
        val_list = val_list + list(np.random.choice(leftover_list, testval_number, replace=False))
        leftover_list = [filepath for filepath in leftover_list if filepath not in val_list]
        if max_train_samples and len(leftover_list)>max_train_samples:
            train_list = train_list + list(np.random.choice(leftover_list, max_train_samples, replace=False))
            train_number = max_train_samples
        else:
            train_list = train_list + leftover_list
            train_number = len(leftover_list)
        print('%d: %d train, %d val, %d test' % (c, train_number, testval_number, testval_number))
    print('Total: %d train, %d val, %d test' % (len(train_list),len(val_list),len(test_list)))

    # Shuffle before returning for good measure
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)

    return train_list, val_list, test_list
