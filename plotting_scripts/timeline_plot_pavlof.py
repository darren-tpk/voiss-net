# Import dependencies
from obspy import UTCDateTime
from toolbox import generate_timeline_indicators, plot_timeline

# Arguments for generate_timeline_indicators
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PV6,PVV'
station1 = 'PN7A,PS1A,PS4A,PV6,PVV'
station2 = 'PN7A,PS1A,PS4A,PV6A,PVV'
location = ''
model_path = './models/4min_all_augmented_revised_model.h5'
meanvar_path = './models/4min_all_augmented_revised_meanvar.npy'
overlap = 0.25
spec_kwargs = None

# Arguments for plot_timeline
type = 'seismic'
channel = '*HZ'

# Plot timeline for 2007
starttime = UTCDateTime(2007,8,1)
endtime = UTCDateTime(2007,11,1)
time_step = 60
npy_dir = None
indicators_path = './pavlof_2007_indicators.pkl'
plot_path = './pavlof_2007.png'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
# generate_timeline_indicators(source, network, station, channel, location, starttime, endtime, model_path, meanvar_path,
#                              overlap, npy_dir, spec_kwargs, export_path=indicators_path)
plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title, export_path=plot_path,
              transparent=False, plot_stations=station, plot_labels=False, labels_kwargs=None)

# Plot timeline for 2013
starttime = UTCDateTime(2013,2,13)
endtime = UTCDateTime(2013,9,30)
time_step = 60
npy_dir = None
indicators_path = './pavlof_2013_indicators.pkl'
plot_path = './pavlof_2013.png'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
# generate_timeline_indicators(source, network, station1, channel, location, starttime, endtime, model_path, meanvar_path,
#                              overlap, npy_dir, spec_kwargs, export_path=indicators_path)
plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title, export_path=plot_path,
              transparent=False, plot_stations=station1, plot_labels=False, labels_kwargs=None)

# Plot timeline for 2014
starttime = UTCDateTime(2014,8,13)
endtime = UTCDateTime(2015,2,26)
time_step = 60
npy_dir = None
indicators_path = './pavlof_2014_indicators.pkl'
plot_path = './pavlof_2014.png'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
# generate_timeline_indicators(source, network, station1, channel, location, starttime, endtime, model_path, meanvar_path,
#                              overlap, npy_dir, spec_kwargs, export_path=indicators_path)
plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title, export_path=plot_path,
              transparent=False, plot_stations=station1, plot_labels=False, labels_kwargs=None)

# Plot timeline for 2016
starttime = UTCDateTime(2015,12,28)
endtime = UTCDateTime(2016,11,4)
time_step = 60
npy_dir = None
indicators_path = './pavlof_2016_indicators.pkl'
plot_path = './pavlof_2016.png'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
# generate_timeline_indicators(source, network, station1, channel, location, starttime, endtime, model_path, meanvar_path,
#                              overlap, npy_dir, spec_kwargs, export_path=indicators_path)
plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title, export_path=plot_path,
              transparent=False, plot_stations=station1, plot_labels=False, labels_kwargs=None)

# Plot timeline for 2021-2022 (seismic)
starttime = UTCDateTime(2022,3,25)
endtime = UTCDateTime(2023,3,1)
time_step = 60
npy_dir = None
indicators_path = './pavlof_2021_indicators.pkl'
plot_path = './pavlof_2021.png'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
generate_timeline_indicators(source, network, station2, channel, location, starttime, endtime, model_path, meanvar_path,
                             overlap, npy_dir, spec_kwargs, export_path=indicators_path)
plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title, export_path=plot_path,
              transparent=False, plot_stations=station2, plot_labels=False, labels_kwargs=None)

# Plot timeline for 2021-2022 (infrasound)
type = 'infrasound'
channel = '*DF'
model_path = './models/4min_all_augmented_infra_revised_model.h5'
meanvar_path = './models/4min_all_augmented_infra_revised_meanvar.npy'
starttime = UTCDateTime(2021,1,1)
endtime = UTCDateTime(2023,3,1)
time_step = 60
npy_dir = None # '/home/ptan/tremor_ml/all_npys/pavlof_2021_2023_infra_npy/'
indicators_path = './pavlof_2021_infra_indicators.pkl'
plot_path = './pavlof_2021_infra.png'
plot_title = 'Timeline for Pavlof Infrasound Tremor (%s to %s)' % (starttime.strftime('%Y-%m-%d'),endtime.strftime('%Y-%m-%d'))
generate_timeline_indicators(source, network, station2, channel, location, starttime, endtime, model_path, meanvar_path,
                             overlap, npy_dir, spec_kwargs, export_path=indicators_path)
plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title, export_path=plot_path,
              transparent=False, plot_stations=station2, plot_labels=False, labels_kwargs=None)