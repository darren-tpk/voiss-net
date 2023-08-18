# Import dependencies
from obspy import UTCDateTime
from toolbox import check_timeline, check_timeline2

# Define variables for timeline checker function
source = 'IRIS'
network = 'AV'
station = 'SSBA,SSLS'
location = ''
seis_channel = '*HZ'
infra_channel = 'BDF'
starttime = UTCDateTime(2023, 8, 15, 10)
endtime = starttime + 3*3600
# seis_model_path = './models/4min_all_subsampled2_model.h5'
# seis_meanvar_path = './models/4min_all_subsampled2_meanvar.npy'
seis_model_path = './models/4min_all_augmented_model.h5'
seis_meanvar_path = './models/4min_all_augmented_meanvar.npy'
# infra_model_path = './models/4min_all_infra_subsampled0_model.h5'
# infra_meanvar_path = './models/4min_all_infra_subsampled0_meanvar.npy'
infra_model_path = './models/4min_all_augmented_infra_model.h5'
infra_meanvar_path = './models/4min_all_augmented_infra_meanvar.npy'
npy_dir = None
fig_width = 12
FONT_S = 12
class_cbar = True
spec_kwargs = None
annotate = True
export_path = './figures/'
transparent = None

# Check timeline for seismic
# predmat = check_timeline(source, network, station, seis_channel, location, starttime, endtime,
#                          seis_model_path, seis_meanvar_path, npy_dir=npy_dir, fig_width=fig_width,
#                          spec_kwargs=spec_kwargs, class_cbar=class_cbar, annotate=annotate,
#                          export_path=export_path, transparent=transparent)
predmat = check_timeline2(source, network, station, seis_channel, location, starttime, endtime,
                         seis_model_path, seis_meanvar_path, npy_dir=npy_dir, fig_width=fig_width,
                         font_s=FONT_S,spec_kwargs=spec_kwargs, class_cbar=class_cbar,
                         export_path=export_path, transparent=transparent)

starttime = UTCDateTime(2021, 8, 6, 10, 00)
endtime = starttime + 3*3600

# Check timeline for infrasound
# predmat = check_timeline(source, network, station, infra_channel, location, starttime, endtime,
#                          infra_model_path, infra_meanvar_path, npy_dir=npy_dir, fig_width=fig_width,
#                          spec_kwargs=spec_kwargs, class_cbar=class_cbar, annotate=annotate,
#                          export_path=export_path, transparent=transparent)
predmat = check_timeline2(source, network, station, infra_channel, location, starttime, endtime,
                         infra_model_path, infra_meanvar_path, npy_dir=npy_dir, fig_width=fig_width,
                         spec_kwargs=spec_kwargs, class_cbar=class_cbar,
                         export_path=export_path, transparent=transparent)
