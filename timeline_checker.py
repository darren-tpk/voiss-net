from obspy import UTCDateTime
from toolbox import check_timeline

source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
location = ''
channel = '*HZ'
starttime = UTCDateTime(2021, 8, 6, 10, 0, 0)
endtime = starttime + 2*3600
model_path = './models/4min_all_subsampled2_model.h5'
npy_dir = None
export_path = None

check_timeline(source, network, station, channel, location, starttime, endtime,
               model_path, npy_dir=npy_dir, export_path=export_path)

channel = '*DF'
model_path = './models/4min_all_infra_subsampled0_model.h5'

check_timeline(source, network, station, channel, location, starttime, endtime,
               model_path, npy_dir=npy_dir, export_path=export_path)