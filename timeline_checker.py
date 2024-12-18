# Import dependencies
from obspy import UTCDateTime
from toolbox import check_timeline, sort_sta_distance

# Define variables for timeline checker function
OVERLAP = 0.5  # 1 min time step for 1 min interval
GENERATE_FIG = True
FIG_WIDTH = 8
FIG_HEIGHT = 6
FONT_S = 8
MODEL_PATH = './models/voissnet_seismic_generalized_model.keras'
MEANVAR_PATH = ''
PNORM_THRESH = 0.4  # threshold for p-norm, can be None
SPEC_KWARGS = None
EXPORT_PATH = None
TRANSPARENT = None

# Check timeline for seismic
SOURCE = 'IRIS'
NETWORK = 'AV'
STATION = 'CERB,CESW,CEPE,CETU,CEAP'
LOCATION = ''
CHANNEL = 'BHZ'
STARTTIME = UTCDateTime(2021, 7, 27, 7, 30)
ENDTIME = STARTTIME + 3*3600

# set up the DR kwargs, can leave as None if not using DR
DR_KWARGS = {'reference_station': 'CERB',       # station code
             'filter_band': (1, 10),            # Hz
             'window_length': 120,              # seconds
             'overlap': 0.5,                    # fraction of window length
             'volc_lat': 51.926630,             # decimal degrees
             'volc_lon': 179.591230,            # decimal degrees
             'seis_vel': 1500,                  # m/s
             'dominant_freq': 2}                # Hz

# sort the stations by distance from the volcano
STA_SORT, DIST_SORT = sort_sta_distance(SOURCE, NETWORK, STATION, STARTTIME,
                                        ENDTIME, CHANNEL, DR_KWARGS)


# now create the timeline plot
class_mat, prob_mat = check_timeline(SOURCE, NETWORK, STA_SORT, CHANNEL,
                                     LOCATION, STARTTIME, ENDTIME, MODEL_PATH,
                                     MEANVAR_PATH, OVERLAP,
                                     pnorm_thresh=PNORM_THRESH,
                                     generate_fig=GENERATE_FIG,
                                     fig_width=FIG_WIDTH,
                                     fig_height=FIG_HEIGHT, font_s=FONT_S,
                                     spec_kwargs=SPEC_KWARGS,
                                     dr_kwargs=DR_KWARGS,
                                     export_path=EXPORT_PATH,
                                     transparent=TRANSPARENT)
