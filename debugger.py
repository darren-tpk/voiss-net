# Plot stats

labels = ['Broadband Tremor',
          'Harmonic Tremor',
          'Monochromatic Tremor',
          'Non-tremor Signal',
          'Explosion',
          'Noise']
stations = ['PN7A','PS1A','PS4A','PV6A','PVV']

# Import all dependencies
import json
import numpy as np
from obspy import UTCDateTime

# Define filepaths and variables for functions
json_filepath = '/Users/darrentpk/Desktop/GitHub/tremor_ml/labels.json'

# Parse json file from label studio
f = open(json_filepath)
labeled_images = json.load(f)
f.close()

# Initialize time bound list
time_bounds = []

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
        continue
    # Otherwise define original width and height of image in pixels and determine pixels indicating each station
    else:
        time_per_percent = (t2 - t1) / 100
        y_span = annotations[0]['original_height']
        y_per_percent = y_span / 100
        station_indicators = np.arange(y_span / (len(stations) * 2), y_span, y_span / (len(stations)))

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


time_bounds = np.array(time_bounds)

all_stations = time_bounds[:,0]
all_durations = (time_bounds[:,2] - time_bounds[:,1]) / 60
all_labels = time_bounds[:,3]

# Report stats
print('No. of time bounds at each station:')
for station in stations:
    print(station, ':', list(all_stations).count(station))

# Stats relating to each label
import numpy as np
for label in labels:
    print('For label: %s' % label)
    indices = np.flatnonzero(all_labels == label)
    print('Number of time bounds: %d' % len(indices))
    print('Average Duration: %.3f min' % np.mean(all_durations[indices]))
    print('Minimum Duration: %.3f min' % np.min(all_durations[indices]))
    print('Maximum Duration: %.3f min' % np.max(all_durations[indices]))
