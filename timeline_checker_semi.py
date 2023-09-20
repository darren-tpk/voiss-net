# Import dependencies
from obspy import UTCDateTime
from toolbox import check_timeline, check_timeline2

# Define variables for timeline checker function
source = 'IRIS'
network = 'AV'
station = 'CERB,CESW,CEPE,CETU,CEAP,CERA'
location = ''
channel = '*HE'
starttime = UTCDateTime(2021, 7, 30, 8)
endtime = starttime + 6*3600
model_path = './models/4min_all_augmented_semi_model.h5'
meanvar_path = './models/4min_all_augmented_semi_meanvar.npy'
npy_dir = None
generate_fig = True
fig_width = 38
class_cbar = True
spec_kwargs = {'freq_lims': (0.5,20)}
# annotate = True
export_path = None

# Check timeline for seismic
class_mat, prob_mat = check_timeline2(source, network, station, channel, location, starttime, endtime,
                                      model_path, meanvar_path, npy_dir=npy_dir, generate_fig=generate_fig,
                                      fig_width=fig_width, class_cbar=class_cbar, spec_kwargs=spec_kwargs,
                                      export_path=export_path)
