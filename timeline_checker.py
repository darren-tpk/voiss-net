# Import dependencies
from obspy import UTCDateTime
from toolbox import check_timeline

# Define variables for timeline checker function
source = 'IRIS'
network = 'AV'
station = 'CERB,CESW,CEPE,CETU,CEAP,CERA'
location = ''
channel = '*HE'

# starttime = UTCDateTime(2021, 9, 14, 16, 0, 0)
# starttime = UTCDateTime(2023, 3, 21, 13)
# endtime = starttime + 3*3600
starttime = UTCDateTime(2021, 7, 30, 8)
endtime = starttime + 3*3600
model_path = './models/4min_all_subsampled2_model.h5'
meanvar_path = './models/4min_all_subsampled2_meanvar.npy'
npy_dir = None
spec_kwargs = None
annotate = True
export_path = None

# Check timeline
predmat = check_timeline(source, network, station, channel, location, starttime, endtime,
                         model_path, meanvar_path, npy_dir=npy_dir, spec_kwargs=spec_kwargs,
                         annotate=annotate, export_path=export_path)

# # Switch to infrasound
# starttime = UTCDateTime(2021, 8, 6, 10, 0, 0)
# endtime = starttime + 3*3600
# channel = '*DF'
# model_path = './models/4min_all_infra_subsampled0_model.h5'
# meanvar_path = './models/4min_all_infra_subsampled0_meanvar.npy'
#
# # Check timeline
# predmat = check_timeline(source, network, station, channel, location, starttime, endtime,
#                          model_path, meanvar_path, npy_dir=npy_dir, annotate=annotate, export_path=export_path)
