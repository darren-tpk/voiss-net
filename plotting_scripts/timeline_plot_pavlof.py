# Import dependencies
from obspy import UTCDateTime
from toolbox import generate_timeline_indicators, plot_timeline

# Arguments for generate_timeline_indicators
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PV6,PVV'
station1 = 'PN7A,PS1A,PS4A,PV6,PVV'
station2 = 'PN7A,PS1A,PS4A,PV6A,PVV'
channel = '*HZ'
location = ''
model_path = './models/4min_all_augmented_new_model.h5'
meanvar_path = './models/4min_all_augmented_new_meanvar.npy'
spec_kwargs = None

# Arguments for plot_timeline
type = 'seismic'

# Plot timeline for 2007
starttime = UTCDateTime(2007,8,1)
endtime = UTCDateTime(2007,11,1)
time_step = 60
npy_dir = None
indicators_path = './pavlof_2007_indicators.pkl'
plot_path = './pavlof_2007.png'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
generate_timeline_indicators(source, network, station, channel, location, starttime, endtime, model_path, meanvar_path,
                             npy_dir, spec_kwargs, export_path=indicators_path)
plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title, export_path=plot_path,
              transparent=False, plot_labels=False, labels_kwargs=None)

# Plot timeline for 2013
starttime = UTCDateTime(2013,2,13)
endtime = UTCDateTime(2013,9,30)
time_step = 240
npy_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2013_npy/'
indicators_path = './pavlof_2013_indicators.pkl'
plot_path = './pavlof_2013.png'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
generate_timeline_indicators(source, network, station, channel, location, starttime, endtime, model_path, meanvar_path,
                             npy_dir, spec_kwargs, export_path=indicators_path)
plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title, export_path=plot_path,
              transparent=False, plot_labels=False, labels_kwargs=None)

# Plot timeline for 2014
starttime = UTCDateTime(2014,8,13)
endtime = UTCDateTime(2015,2,26)
time_step = 240
npy_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2014_npy_2/'
indicators_path = './pavlof_2014_indicators.pkl'
plot_path = './pavlof_2014.png'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
generate_timeline_indicators(source, network, station, channel, location, starttime, endtime, model_path, meanvar_path,
                             npy_dir, spec_kwargs, export_path=indicators_path)
plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title, export_path=plot_path,
              transparent=False, plot_labels=False, labels_kwargs=None)

# Plot timeline for 2016
starttime = UTCDateTime(2015,12,28)
endtime = UTCDateTime(2016,11,4)
time_step = 240
npy_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2016_npy/'
indicators_path = './pavlof_2016_indicators.pkl'
plot_path = './pavlof_2016.png'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
generate_timeline_indicators(source, network, station, channel, location, starttime, endtime, model_path, meanvar_path,
                             npy_dir, spec_kwargs, export_path=indicators_path)
plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title, export_path=plot_path,
              transparent=False, plot_labels=False, labels_kwargs=None)

# Plot timeline for 2021-2022
starttime = UTCDateTime(2021,1,1)
endtime = UTCDateTime(2023,3,1)
time_step = 60
# npy_dir = '/home/ptan/tremor_ml/all_npys/pavlof_2021_2023_npy/'
# npy_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2021_2023_npy/'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
export_path = '/Users/darrentpk/Desktop/pavlof2021_test.png'
plot_labels = True
labels_kwargs = {'start_date': UTCDateTime(2021, 7, 22),
                 'end_date': UTCDateTime(2021, 9, 22),
                 'labels_dir': '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/'}
