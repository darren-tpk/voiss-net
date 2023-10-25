# Import dependencies
from obspy import UTCDateTime
from toolbox import plot_timeline

# Define variables for timeline plotting function
type = 'seismic'
time_step = 240
model_path = './models/4min_all_augmented_new_model.h5'
meanvar_path = './models/4min_all_augmented_new_meanvar.npy'

# Plot timeline for Cleveland
start_month = UTCDateTime(2014,7,25)
end_month = UTCDateTime(2020,9,17)
npy_dir = '/Users/darrentpk/Desktop/all_npys/cleveland_npy/'
plot_title = 'Timeline for Cleveland Seismic Tremor (%s to %s)' % (start_month.strftime('%Y-%m-%d'),end_month.strftime('%Y-%m-%d'))
export_path = '/Users/darrentpk/Desktop/Github/tremor_ml/figures/cleveland.png'
plot_timeline(start_month, end_month, time_step, type, model_path, meanvar_path, npy_dir, plot_title,
              export_path=export_path, transparent=False)

# Plot timeline for Shishaldin 2014
start_month = UTCDateTime(2014,7,1)
end_month = UTCDateTime(2015,12,20)
npy_dir = '/Users/darrentpk/Desktop/all_npys/shishaldin_npy1/'
plot_title = 'Timeline for Shishaldin Seismic Tremor (%s to %s)' % (start_month.strftime('%Y-%m-%d'),end_month.strftime('%Y-%m-%d'))
export_path = '/Users/darrentpk/Desktop/Github/tremor_ml/figures/shishaldin2014_new.png'
plot_timeline(start_month, end_month, time_step, type, model_path, meanvar_path, npy_dir, plot_title,
              export_path=export_path, transparent=False)

# Plot timeline for Shishaldin 2019
start_month = UTCDateTime(2019,4,12)
end_month = UTCDateTime(2020,7,16)
npy_dir = '/Users/darrentpk/Desktop/all_npys/shishaldin_npy2/'
plot_title = 'Timeline for Shishaldin Seismic Tremor (%s to %s)' % (start_month.strftime('%Y-%m-%d'),end_month.strftime('%Y-%m-%d'))
export_path = '/Users/darrentpk/Desktop/Github/tremor_ml/figures/shishaldin2019.png'
plot_timeline(start_month, end_month, time_step, type, model_path, meanvar_path, npy_dir, plot_title,
              export_path=export_path, transparent=False)

# Plot timeline for Okmok 2008
start_month = UTCDateTime(2008,4,12)
end_month = UTCDateTime(2008,11,27)
npy_dir = '/Users/darrentpk/Desktop/all_npys/okmok_npy/'
plot_title = 'Timeline for Okmok Seismic Tremor (%s to %s)' % (start_month.strftime('%Y-%m-%d'),end_month.strftime('%Y-%m-%d'))
export_path = '/Users/darrentpk/Desktop/Github/tremor_ml/figures/okmok2008_new.png'
plot_timeline(start_month, end_month, time_step, type, model_path, meanvar_path, npy_dir, plot_title,
              export_path=export_path, transparent=False)

# Plot timeline for Veniaminof
start_month = UTCDateTime(2004,1,19)
end_month = UTCDateTime(2007,7,26)
npy_dir = '/Users/darrentpk/Desktop/all_npys/veniaminof_npy/'
plot_title = 'Timeline for Veniaminof Seismic Tremor (%s to %s)' % (start_month.strftime('%Y-%m-%d'),end_month.strftime('%Y-%m-%d'))
export_path = '/Users/darrentpk/Desktop/Github/tremor_ml/figures/veniaminof2004.png'
plot_timeline(start_month, end_month, time_step, type, model_path, meanvar_path, npy_dir, plot_title,
              export_path=export_path, transparent=False)