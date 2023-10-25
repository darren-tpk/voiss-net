# %% CREATE LABELED DATASET

# Import all dependencies
import json
import numpy as np
import statistics as st
from obspy import UTCDateTime
from waveform_collection import gather_waveforms
from toolbox import process_waveform, calculate_spectrogram

# Define filepaths and variables for functions
json_filepath = '/Users/darrentpk/Desktop/GitHub/tremor_ml/labels/labels_20230215.json'
time_step = 4 * 60  # Create a training dataset with 2D matrices spanning 4 minutes each
output_dir = '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/'
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
location = ''
channel = '*HZ'
pad = 240  # padding length [s]
client = 'IRIS'  # FDSN client for data pull
filter_band = None  # frequency band for bandpass filter
window_duration = 10  # spectrogram window duration [s]
freq_lims = (0.5, 10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive
demean = False  # remove the temporal mean of the plotted time span from the spectrogram matrix

# Create label dictionary
label_dict = {'Broadband Tremor': 0,
              'Harmonic Tremor': 1,
              'Monochromatic Tremor': 2,
              'Non-tremor Signal': 3,
              'Explosion': 4,
              'Noise': 5}

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
                                      taper_percentage=None, filter_band=filter_band, verbose=False)
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
        spec_db, utc_times = calculate_spectrogram(trace, t1, t2, window_duration, freq_lims, demean=demean)

        # Get label time bounds that are observed on station
        time_bounds_station = [tb for tb in time_bounds if tb[0] == stream_station]

        # Define array of time steps for spectrogram slicing
        step_bounds = np.arange(t1, t2+time_step, time_step)

        # Loop over time steps
        for j in range(len(step_bounds) - 1):

            # Slice spectrogram
            sb1 = step_bounds[j]
            sb2 = step_bounds[j + 1]
            spec_slice_indices = np.flatnonzero([sb1 < t < sb2 for t in utc_times])
            spec_slice = spec_db[:, spec_slice_indices]

            # Enforce shape
            if np.shape(spec_slice) != (94, time_step):
                # Try inclusive slicing time span (<= sb2)
                spec_slice_indices = np.flatnonzero([sb1 < t <= sb2 for t in utc_times])
                spec_slice = spec_db[:, spec_slice_indices]
                # If it still doesn't fit our shape, try double inclusive time bounds
                if np.shape(spec_slice) != (94, time_step):
                    spec_slice_indices = np.flatnonzero([sb1 <= t <= sb2 for t in utc_times])
                    spec_slice = spec_db[:, spec_slice_indices]
                    # Otherwise, raise error
                    if np.shape(spec_slice) != (94, time_step):
                        raise ValueError('THE SHAPE IS NOT RIGHT.')

            # Skip matrices that have a spectrogram data gap
            if np.sum(spec_slice.flatten() < -220) > (0.2 * time_step):
                print('Skipping due to data gap, %d elements failed the check' % np.sum(spec_slice.flatten() < -220))
                continue

            # Obtain corresponding time samples for spectrogram slice
            time_slice = utc_times[spec_slice_indices]

            # Check for overlaps and fill a vector with labels to decide final label
            label_indices = np.ones(len(time_slice)) * -1
            for time_bound_station in time_bounds_station:

                # Labeled time bound starts in slice and ends in slice
                if time_bound_station[1] >= sb1 and time_bound_station[2] <= sb2:
                    valid_indices = np.flatnonzero(np.logical_and(time_slice >= time_bound_station[1], time_slice <= time_bound_station[2]))
                    label_indices[valid_indices] = label_dict[time_bound_station[3]]

                # Labeled time bound starts before slice and ends after slice
                elif time_bound_station[1] < sb1 and time_bound_station[2] > sb2:
                    label_indices[:] = label_dict[time_bound_station[3]]

                # Labeled time bound starts before slice and ends in slice
                elif time_bound_station[1] < sb1 and (sb1 <= time_bound_station[2] <= sb2):
                    label_indices[np.flatnonzero(time_slice <= time_bound_station[2])] = label_dict[time_bound_station[3]]

                # Labeled time bound starts in slice and ends after slice
                elif (sb1 <= time_bound_station[1] <= sb2) and time_bound_station[2] > sb2:
                    label_indices[np.flatnonzero(time_slice >= time_bound_station[1])] = label_dict[time_bound_station[3]]

            # Count how many time samples correspond to each label
            labels_seen, label_counts = np.unique(label_indices, return_counts=True)

            # If it's all unlabeled, skip
            if len(labels_seen) == 1 and -1 in labels_seen:
                continue
            # If we see the explosion label in >10% of the time samples, choose explosion
            elif label_dict['Explosion'] in labels_seen and label_counts[list(labels_seen).index(label_dict['Explosion'])] >= 0.1*len(label_indices):
                final_label = 'Explosion'
            # If we see the non-tremor signal label in >10% of the time samples, choose non-tremor signal
            elif label_dict['Non-tremor Signal'] in labels_seen and label_counts[list(labels_seen).index(label_dict['Non-tremor Signal'])] >= 0.1 * len(label_indices):
                final_label = 'Non-tremor Signal'
            # If we don't see either, and less than 50% of the time samples are meaningfully labeled, skip
            elif np.max(label_counts) <= 0.5 * len(label_indices) or st.mode(label_indices) == -1:
                continue
            # If we see a tremor or noise label in > 50% of the time samples, choose that label
            else:
                final_label_value = int(labels_seen[np.argmax(label_counts)])
                final_label = next(key for key, value in label_dict.items() if value == final_label_value)

            file_name = stream_station + '_' + sb1.strftime('%Y%m%d%H%M') + '_' + \
                        sb2.strftime('%Y%m%d%H%M') + '_' + str(label_dict[final_label]) + '.npy'
            np.save(output_dir + file_name, spec_slice)
