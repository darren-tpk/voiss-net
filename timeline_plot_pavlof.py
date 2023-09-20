# Import dependencies
from obspy import UTCDateTime
from toolbox import plot_timeline

# Define variables for timeline plotting function
type = 'seismic'
time_step = 240
model_path = './models/4min_all_augmented_drop_model.h5'
meanvar_path = './models/4min_all_augmented_drop_meanvar.npy'

# Plot timeline for 2013
start_month = UTCDateTime(2013,2,13)
end_month = UTCDateTime(2013,10,3)
npy_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2013_npy/'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (start_month.strftime('%Y-%m-%d'),end_month.strftime('%Y-%m-%d'))
export_path = '/Users/darrentpk/Desktop/pavlof2013x.png'
plot_timeline(start_month, end_month, time_step, type, model_path, meanvar_path, npy_dir, plot_title,
              export_path=export_path, transparent=False)

# Plot timeline for 2014
start_month = UTCDateTime(2014,8,13)
end_month = UTCDateTime(2015,2,26)
npy_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2014_npy_2/'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (start_month.strftime('%Y-%m-%d'),end_month.strftime('%Y-%m-%d'))
export_path = '/Users/darrentpk/Desktop/pavlof2014.png'
plot_timeline(start_month, end_month, time_step, type, model_path, meanvar_path, npy_dir, plot_title,
              export_path=export_path, transparent=False)

# Plot timeline for 2016
start_month = UTCDateTime(2015,12,28)
end_month = UTCDateTime(2016,11,4)
npy_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2016_npy/'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (start_month.strftime('%Y-%m-%d'),end_month.strftime('%Y-%m-%d'))
export_path = '/Users/darrentpk/Desktop/pavlof2016.png'
plot_timeline(start_month, end_month, time_step, type, model_path, meanvar_path, npy_dir, plot_title,
              export_path=export_path, transparent=False)

# Plot timeline for 2021-2022
start_month = UTCDateTime(2021,1,1)
end_month = UTCDateTime(2023,3,1)
time_step = 60
npy_dir = '/Users/darrentpk/Desktop/all_npys/pavlof_2021_2023_npy/'
plot_title = 'Timeline for Pavlof Seismic Tremor (%s to %s)' % (start_month.strftime('%Y-%m-%d'),end_month.strftime('%Y-%m-%d'))
export_path = '/Users/darrentpk/Desktop/pavlof2021.png'
plot_labels = True
labels_kwargs = {'start_date': UTCDateTime(2021, 7, 22),
                 'end_date': UTCDateTime(2021, 9, 22),
                 'labels_dir': '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/'}
plot_timeline(start_month, end_month, time_step, type, model_path, meanvar_path, npy_dir, plot_title,
              export_path=export_path, transparent=False, plot_labels=plot_labels, labels_kwargs=labels_kwargs)

