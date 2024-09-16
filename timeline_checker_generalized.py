# Import dependencies
from obspy import UTCDateTime
from toolbox import check_timeline

# Define variables for timeline checker function
source = 'IRIS'
channel = '*HZ'
location = ''
overlap = 0.5  # 1 min time step for 2 min interval
generate_fig = True
fig_width = 12
fig_height = 10
font_s = 12
spec_kwargs = None
export_path = '/Users/darrentpk/Desktop/'
transparent = None
model_path = './models/voissnet_seismic_2min_20240913_model.keras'
meanvar_path = None
pnorm_thresh = 0.4
dr_kwargs=None

# Example 1: Shishaldin
network = 'AV'
station = 'SSBA,SSLS,SSLN'
starttime = UTCDateTime(2023,10,2,12,0)
endtime = UTCDateTime(2023,10,4,12,0)
class_mat, prob_mat = check_timeline(source, network, station, channel, location, starttime, endtime,
                                     model_path, meanvar_path, overlap, pnorm_thresh=pnorm_thresh, generate_fig=generate_fig,
                                     fig_width=fig_width, fig_height=fig_height, font_s=font_s, spec_kwargs=spec_kwargs,
                                     dr_kwargs=dr_kwargs, export_path=export_path, transparent=transparent)
# Example 2: Kilauea
network = 'HV'
station = 'UWE,UWB,WRM,HAT'
starttime = UTCDateTime(2023,6,7,14,0)
endtime = UTCDateTime(2023,6,7,17,0)
class_mat, prob_mat = check_timeline(source, network, station, channel, location, starttime, endtime,
                                     model_path, meanvar_path, overlap, pnorm_thresh=pnorm_thresh, generate_fig=generate_fig,
                                     fig_width=fig_width, fig_height=fig_height, font_s=font_s, spec_kwargs=spec_kwargs,
                                     dr_kwargs=dr_kwargs, export_path=export_path, transparent=transparent)

# Example 3: Pavlof
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
starttime = UTCDateTime(2021,3,18,5,0)
endtime = UTCDateTime(2021,3,18,8,0)
class_mat, prob_mat = check_timeline(source, network, station, channel, location, starttime, endtime,
                                     model_path, meanvar_path, overlap, pnorm_thresh=pnorm_thresh, generate_fig=generate_fig,
                                     fig_width=fig_width, fig_height=fig_height, font_s=font_s, spec_kwargs=spec_kwargs,
                                     dr_kwargs=dr_kwargs, export_path=export_path, transparent=transparent)

# Example 4: Great Sitkin
network = 'AV'
station = 'GSMY,GSTD,GSTR,GSSP,GSCK'
starttime = UTCDateTime(2021,5,26,4,0)
endtime = UTCDateTime(2021,5,26,5,30)
class_mat, prob_mat = check_timeline(source, network, station, channel, location, starttime, endtime,
                                     model_path, meanvar_path, overlap, pnorm_thresh=pnorm_thresh, generate_fig=generate_fig,
                                     fig_width=fig_width, fig_height=fig_height, font_s=font_s, spec_kwargs=spec_kwargs,
                                     dr_kwargs=dr_kwargs, export_path=export_path, transparent=transparent)

# Example 5: Semisopochnoi
network = 'AV'
station = 'CERB,CESW,CEPE,CETU,CEAP,CERA'
starttime = UTCDateTime(2021,7,30,8,0)
endtime = UTCDateTime(2021,7,30,11,0)
class_mat, prob_mat = check_timeline(source, network, station, channel, location, starttime, endtime,
                                     model_path, meanvar_path, overlap, pnorm_thresh=pnorm_thresh, generate_fig=generate_fig,
                                     fig_width=fig_width, fig_height=fig_height, font_s=font_s, spec_kwargs=spec_kwargs,
                                     dr_kwargs=dr_kwargs, export_path=export_path, transparent=transparent)

# Example 6: Trident
network = 'AV'
station = 'KBM,KABU,KAKN,KVT'
starttime = UTCDateTime(2023,12,4,4,45)
endtime = UTCDateTime(2023,12,4,7,15)
class_mat, prob_mat = check_timeline(source, network, station, channel, location, starttime, endtime,
                                     model_path, meanvar_path, overlap, pnorm_thresh=pnorm_thresh, generate_fig=generate_fig,
                                     fig_width=fig_width, fig_height=fig_height, font_s=font_s, spec_kwargs=spec_kwargs,
                                     dr_kwargs=dr_kwargs, export_path=export_path, transparent=transparent)
