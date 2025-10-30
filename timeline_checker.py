# Import dependencies
from obspy import UTCDateTime, Stream
from obspy.geodetics import gps2dist_azimuth
from waveform_collection import gather_waveforms
from toolbox import check_timeline

# Define variables for data pull
SOURCE = "IRIS"
NETWORK = "AV"
STATION = "CERB,CESW,CEPE,CETU,CEAP"
LOCATION = ""
CHANNEL = "BHZ"
STARTTIME = UTCDateTime(2021, 7, 27, 7, 30)
ENDTIME = STARTTIME + 3*3600
PAD = 360  # s
N_JOBS = 1  # number of parallel jobs for data pull, can be -1 for all cores

# Define variables for check_timeline function
OVERLAP = 0.5  # 1 min time step for 1 min interval
GENERATE_FIG = True
FIG_WIDTH = 8
FIG_HEIGHT = 6
FONT_S = 8
MODEL_PATH = "./models/voissnet_seismic_generalized_model.keras"
MEANVAR_PATH = ""
PNORM_THRESH = 0.4  # threshold for p-norm, can be None
SPEC_KWARGS = None
EXPORT_PATH = None
TRANSPARENT = None

# [Reduced Displacement] Set up the DR kwargs, can leave as None if not using DR
DR_STATION = STATION.split(",")[0]
VOLC_COORDS = (51.926630, 179.591230)
DR_KWARGS = {"reference_station": DR_STATION,   # station code
             "filter_band": (1, 10),            # Hz
             "window_length": 120,              # seconds
             "overlap": 0.5,                    # fraction of window length
             "volc_lat": VOLC_COORDS[0],        # decimal degrees
             "volc_lon": VOLC_COORDS[1],        # decimal degrees
             "seis_vel": 1500,                  # m/s
             "dominant_freq": 2}                # Hz

# [Frequency Index] Set up FI kwargs as an alternative to DR kwargs
FI_KWARGS = {"reference_station": "all",        # station code or "all"
             "window_length": 10,               # seconds
             "overlap": 0.5,                    # fraction of window length
             "filomin": 1,                      # Hz -- FI lower band minimum
             "filomax": 2.5,                    # Hz -- FI lower band maximum
             "fiupmin": 5,                      # Hz -- FI upper band minimum
             "fiupmax": 10,                     # Hz -- FI upper band maximum
             "med_filt_kernel": None,           # Kernel size for median filter smoothing
             "volc_lat": VOLC_COORDS[0],        # decimal degrees (only for source-station distance in y-label)
             "volc_lon": VOLC_COORDS[1]}        # decimal degrees (only for source-station distance in y-label)

# Pull data
stream = gather_waveforms(SOURCE,
                          NETWORK,
                          STATION,
                          LOCATION,
                          CHANNEL,
                          STARTTIME - PAD,
                          ENDTIME + PAD,
                          n_jobs=N_JOBS)

# Sort stream by distance to volcano (order of the stream input determines order of subplots)
stream.traces.sort(key=lambda tr: gps2dist_azimuth(VOLC_COORDS[0], VOLC_COORDS[1], tr.stats.latitude, tr.stats.longitude)[0])

# Run VOISS-Net
class_mat, prob_mat = check_timeline(stream,
                                     STARTTIME,
                                     ENDTIME,
                                     MODEL_PATH,
                                     MEANVAR_PATH,
                                     OVERLAP,
                                     pnorm_thresh=PNORM_THRESH,
                                     generate_fig=GENERATE_FIG,
                                     fig_width=FIG_WIDTH,
                                     fig_height=FIG_HEIGHT,
                                     font_s=FONT_S,
                                     spec_kwargs=SPEC_KWARGS,
                                     dr_kwargs=DR_KWARGS,        # or fi_kwargs=FI_KWARGS
                                     export_path=EXPORT_PATH,
                                     transparent=TRANSPARENT)