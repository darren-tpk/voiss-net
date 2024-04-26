# Import dependencies
from obspy import UTCDateTime
from toolbox import check_timeline

# Define variables for timeline checker function
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
location = ''
overlap = 0.75  # 1 min time step for 4 min interval
generate_fig = True
fig_width = 12
fig_height = 10
font_s = 12
spec_kwargs = None
annotate = False
export_path = None
transparent = None

# Check timeline for seismic
channel = '*HZ'
starttime = UTCDateTime(2021, 9, 14, 16)
endtime = starttime + 3*3600
model_path = './models/voissnet_seismic_model.h5'
meanvar_path = './models/voissnet_seismic_meanvar.npy'
pnorm_thresh = None
dr_kwargs = {'reference_station': 'PS1A',    # station code
             'filter_band': (1, 5),          # Hz
             'window_length': 10,            # seconds
             'overlap': 0.5,                 # fraction of window length
             'volc_lat': 55.4173,            # decimal degrees
             'volc_lon': -161.8937,          # decimal degrees
             'seis_vel': 1500,               # m/s
             'dominant_freq': 2}             # Hz
class_mat, prob_mat = check_timeline(source, network, station, channel, location, starttime, endtime,
                                     model_path, meanvar_path, overlap, pnorm_thresh=pnorm_thresh, generate_fig=generate_fig,
                                     fig_width=fig_width, fig_height=fig_height, font_s=font_s, spec_kwargs=spec_kwargs,
                                     dr_kwargs=dr_kwargs, export_path=export_path, transparent=transparent)

# Check timeline for infrasound
channel = 'BDF'
starttime = UTCDateTime(2021, 8, 6, 10)
endtime = starttime + 3*3600
model_path = './models/voissnet_infrasound_model.h5'
meanvar_path = './models/voissnet_infrasound_meanvar.npy'
pnorm_thresh = None
dr_kwargs = None
class_mat, prob_mat = check_timeline(source, network, station, channel, location, starttime, endtime,
                                     model_path, meanvar_path, overlap, pnorm_thresh=pnorm_thresh, generate_fig=generate_fig,
                                     fig_width=fig_width, fig_height=fig_height, font_s=font_s, spec_kwargs=spec_kwargs,
                                     dr_kwargs=dr_kwargs, export_path=export_path, transparent=transparent)
