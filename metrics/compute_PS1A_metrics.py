from obspy import UTCDateTime, Stream
from waveform_collection import gather_waveforms
from toolbox import compute_metrics, process_waveform, calculate_spectrogram
import numpy as np

# Define variables
time_step = 4 * 60  # s
starttime = UTCDateTime(2021, 1, 1, 0, 0, 0)
endtime = UTCDateTime(2023, 1, 1, 0, 0, 0)
tag = '_20210101_20230101'
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
location = ''
channel = '*HZ'
pad = 360  # padding length [s]
window_duration = 10  # spectrogram window duration [s]
freq_lims = (0.5, 10)  # frequency limits for output spectrogram. If `None`, the limits will be adaptive

# Create function to size variable correct
def correct_dimensions(var, used_stations, desired_dim=(5,119)):
    station_order = 'PN7A,PS1A,PS4A,PV6A,PVV'.split(',')
    base_matrix = np.ones(desired_dim) * np.nan
    for n, used_station in enumerate(used_stations):
        base_matrix[station_order.index(used_station),:] = var[n,:]
    return base_matrix

spec_db_all = []
utc_times_all = []

# Calculate number of time steps for loop
calculation_step = 4*60*60
num_steps = int((endtime-starttime)/(4*60*60))

# Loop over days
for i in range(num_steps):

    # Mark time
    t1 = starttime + (i*calculation_step)
    t2 = t1 + calculation_step
    print('Now at %s ...' % t1)

    # Load data using waveform_collection tool
    successfully_computed = False
    while not successfully_computed:
        try:
            stream = gather_waveforms(source=source, network=network, station=station, location=location, channel=channel,
                                      starttime=t1 - pad, endtime=t2 + pad, verbose=False)
            tmpl, dr, pe, fc, fd, fsd = compute_metrics(stream, process_taper=60, metric_taper=pad,
                                                          filter_band=(0.5, 10), window_length=time_step, overlap=0.5)
            successfully_computed = True
        except:
            print('gather_waveforms failed to obtain response')
            pass

    # Correct shapes
    stream_stations = [tr.stats.station for tr in stream]
    tmpl = correct_dimensions(tmpl, stream_stations)
    dr = correct_dimensions(dr, stream_stations)
    pe = correct_dimensions(pe, stream_stations)
    fc = correct_dimensions(fc, stream_stations)
    fd = correct_dimensions(fd, stream_stations)
    fsd = correct_dimensions(fsd, stream_stations)

    # Get PS1A trace
    if 'PS1A' in stream_stations:
        PS1A_trace = Stream(stream[stream_stations.index('PS1A')])
        try:
            PS1A_trace = process_waveform(PS1A_trace, remove_response=True, detrend=False, taper_length=pad,
                                      taper_percentage=None, filter_band=None, verbose=False)
        except:
            successfully_processed = False
            while not successfully_processed:
                try:
                    PS1A_trace = gather_waveforms(source=source, network=network, station='PS1A', location=location,
                                              channel=channel, starttime=t1 - pad, endtime=t2 + pad, verbose=False)
                    PS1A_trace = process_waveform(PS1A_trace, remove_response=True, detrend=False, taper_length=pad,
                                              taper_percentage=None, filter_band=None, verbose=False)
                    successfully_processed = True
                except:
                    print('gather_waveforms failed to retrieve response for PS1A')
                    pass

        # Calculate spectrogram
        PS1A_trace = PS1A_trace[0]
        spec_db, utc_times = calculate_spectrogram(PS1A_trace, t1, t2, window_duration, freq_lims, overlap=0)
    else:
        spec_db = np.ones((94, 1440)) * np.nan
        utc_times = np.arange(utc_times_all[-1] + window_duration/2, utc_times_all[-1] + calculation_step, window_duration)

    if i == 0:
        tmpl_all = tmpl
        dr_all = dr
        pe_all = pe
        fc_all = fc
        fd_all = fd
        fsd_all = fsd
        spec_db_all = spec_db
        utc_times_all = utc_times
    else:
        tmpl_all = np.hstack((tmpl_all, tmpl))
        dr_all = np.hstack((dr_all, dr))
        pe_all = np.hstack((pe_all, pe))
        fc_all = np.hstack((fc_all, fc))
        fd_all = np.hstack((fd_all, fd))
        fsd_all = np.hstack((fsd_all, fsd))
        spec_db_all = np.hstack((spec_db_all, spec_db))
        utc_times_all = np.hstack((utc_times_all, utc_times))

metrics_dir = '/Users/darrentpk/Desktop/GitHub/tremor_ml/metrics/PS1A_BHZ_metrics/'
np.save(metrics_dir + 'tmpl_all' + tag + '.npy', tmpl_all)
np.save(metrics_dir + 'dr_all' + tag + '.npy', dr_all)
np.save(metrics_dir + 'pe_all' + tag + '.npy', pe_all)
np.save(metrics_dir + 'fc_all' + tag + '.npy', fc_all)
np.save(metrics_dir + 'fd_all' + tag + '.npy', fd_all)
np.save(metrics_dir + 'fsd_all' + tag + '.npy', fsd_all)
np.save(metrics_dir + 'spec_db_all' + tag + '.npy', spec_db_all)
np.save(metrics_dir + 'utc_times_all' + tag + '.npy', utc_times_all)

