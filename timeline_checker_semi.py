# Import dependencies
from obspy import UTCDateTime
from toolbox import check_timeline

# Define variables for timeline checker function
source = 'IRIS'
network = 'AV'
station = 'CERB,CESW,CEPE,CETU,CEAP,CERA'
location = ''
seis_channel = '*HZ'
starttime = UTCDateTime(2021, 7, 30, 8)
endtime = starttime + 3*3600
seis_model_path = './models/4min_all_subsampled0_model_semi.h5'
seis_meanvar_path = './models/4min_all_subsampled0_meanvar_semi.npy'
npy_dir = None
fig_width = 34.5
class_cbar = True
spec_kwargs = {'freq_lims':(0.5,20)}
annotate = True
export_path = None

# Check timeline for seismic
predmat = check_timeline(source, network, station, seis_channel, location, starttime, endtime,
                         seis_model_path, seis_meanvar_path, npy_dir=npy_dir, fig_width=fig_width,
                         spec_kwargs=spec_kwargs, class_cbar=class_cbar, annotate=annotate, export_path=export_path)
