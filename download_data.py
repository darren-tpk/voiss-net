#%% DOWNLOAD DATA

# This script downloads data from a chosen seismic network across a user-defined duration, using ObsPy's st.read() and st.write().

# Import all dependencies
import glob
import time
import pandas
import numpy as np
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client

#%% Define variables

# Define variables
data_destination = '/home/data/pavlof/'
start_time = UTCDateTime(2021,1,1,0,0,0)
end_time = UTCDateTime(2022,9,1,0,0,0)
station_list_filename = '/home/ptan/tremor_ml/station_csv/pavlof_stations.csv'

#%% Define functions

# Nil

#%% Loop over sta-chan-loc combinations to download data into user-defined destination

# Get list of station info
station_list = pandas.read_csv(station_list_filename)

# Dissect duration into days
num_days = int(np.floor((end_time - start_time) / 86400))
print("\nCommencing data fetch...")
time_start = time.time()

# Commence loop over days
for i in range(num_days):

    # Define temporal boundaries for data fetch
    t1 = start_time + (i * 86400)
    t2 = start_time + ((i + 1) * 86400)
    print('\nNow at %s...' % str(t1.date))

    # Loop over station list and extract relevant seed information
    for j in range(len(station_list.Station)):

        client = Client(station_list.Client[j])
        station = station_list.Station[j]
        network = station_list.Network[j]
        channels = [channel.strip() for channel in station_list.Channels[j].split(',')]
        location = station_list.Location[j]

        # Loop over channels
        for channel in channels:

            # Check if file already exists
            julday_str = '%03d' % t1.julday
            seed_sample_filepath = data_destination + station + '.' + channel + '.' + str(t1.year) + ':' + julday_str + ':*'
            matching_data_files = glob.glob(seed_sample_filepath)
            if len(matching_data_files) != 0:
                print('%s already has data files for %s, skipping.' % (station, t1.date))
                continue

            # Get waveforms by querying client
            try:
                st = client.get_waveforms(network, station, location, channel, t1, t2)

                # Save every trace separately
                for tr in st:
                    # Craft the seed file name
                    trace_year = str(tr.stats.starttime.year)
                    trace_julday = str(tr.stats.starttime.julday)
                    trace_time = str(tr.stats.starttime.time)[0:8]
                    trace_datetime_str = ":".join([trace_year, str(trace_julday).zfill(3), trace_time])
                    seed_filename = station + '.' + channel + '.' + trace_datetime_str

                    # Write to seed file
                    seed_filepath = data_destination + seed_filename
                    tr.write(seed_filepath, format="MSEED")

                    # Print success
                    print('%s successfully saved.' % seed_filename)

            # continue if waveform retrieval fails
            except:
                print('%s.%s failed.' % (station,channel))
                continue

    # Print progression
    time_current = time.time()
    print('Data collection for the day complete. Elapsed time: %.2f hours' % ((time_current - time_start) / 3600))

# Conclude process
time_end = time.time()
print('\nData collection complete. Time taken: %.2f hours' % ((time_end - time_start) / 3600))
