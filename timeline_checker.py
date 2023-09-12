# Import dependencies
from obspy import UTCDateTime
from toolbox import check_timeline, check_timeline2

# Define variables for timeline checker function
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
location = ''
seis_channel = '*HZ'
infra_channel = 'BDF'
starttime = UTCDateTime(2021, 9, 14, 16)
endtime = starttime + 3*3600
seis_model_path = './models/4min_all_augmented_drop_model.h5'
seis_meanvar_path = './models/4min_all_augmented_drop_meanvar.npy'
infra_model_path = './models/4min_all_augmented_infra_model.h5'
infra_meanvar_path = './models/4min_all_augmented_infra_meanvar.npy'
npy_dir = None
fig_width = 34.5
class_cbar = True
spec_kwargs = None
annotate = False
export_path = '/Users/darrentpk/Desktop/'
transparent = None

# # Check timeline for seismic
predmat = check_timeline2(source, network, station, seis_channel, location, starttime, endtime,
                         seis_model_path, seis_meanvar_path, npy_dir=npy_dir, fig_width=fig_width,
                         spec_kwargs=spec_kwargs, class_cbar=class_cbar,
                         export_path=export_path, transparent=transparent)

# Check timeline for infrasound
starttime = UTCDateTime(2021, 8, 6, 10)
endtime = starttime + 3*3600
predmat = check_timeline2(source, network, station, infra_channel, location, starttime, endtime,
                         infra_model_path, infra_meanvar_path, npy_dir=npy_dir, fig_width=fig_width,
                         spec_kwargs=spec_kwargs, class_cbar=class_cbar,
                         export_path=export_path, transparent=transparent)
