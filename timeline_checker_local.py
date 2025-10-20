# Import dependencies
from obspy import UTCDateTime, Stream, read_inventory
from obspy.geodetics import gps2dist_azimuth
from waveform_collection import read_local
from toolbox import check_timeline, sort_sta_distance, download_data

# Define variables for data download & reading local miniseeds
SOURCE = "IRIS"
NETWORK = "AV"
STATION = "CERB,CESW,CEPE,CETU,CEAP"
LOCATION = ""
CHANNEL = "BHZ"
STARTTIME = UTCDateTime(2021, 7, 27, 7, 30)
ENDTIME = STARTTIME + 3*3600
DATA_DIR = './miniseed/'
METADATA_FILEPATH = './miniseed/metadata.xml'
COORD_FILEPATH = './miniseed/coords.json'
N_JOBS = 1  # number of parallel jobs for data pull, can be -1 for all cores

# Additional variables for reading local miniseeds
PAD = 360
PATTERN = '{net}.{sta}.{loc}.{cha}.{year}.{jday}.{hour}.mseed'

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

# Set up the DR kwargs, can leave as None if not using DR
DR_STATION = STATION.split(',')[0]
VOLC_COORDS = (51.926630, 179.591230)
DR_KWARGS = {"reference_station": DR_STATION,   # station code
             "filter_band": (1, 10),            # Hz
             "window_length": 120,              # seconds
             "overlap": 0.5,                    # fraction of window length
             "volc_lat": VOLC_COORDS[0],        # decimal degrees
             "volc_lon": VOLC_COORDS[1],        # decimal degrees
             "seis_vel": 1500,                  # m/s
             "dominant_freq": 2}                # Hz

# Download data (only need to do this once)
download_data(SOURCE,
              NETWORK,
              STATION,
              LOCATION,
              CHANNEL,
              STARTTIME - 3600,
              ENDTIME + 3600,
              data_dir=DATA_DIR,
              metadata_filepath=METADATA_FILEPATH,
              coord_filepath=COORD_FILEPATH,
              n_jobs=N_JOBS)

# Read data from local miniseed directory and sort by station-source distance
stream = read_local(DATA_DIR,
                    COORD_FILEPATH,
                    NETWORK,
                    STATION,
                    LOCATION,
                    CHANNEL,
                    STARTTIME - PAD,
                    ENDTIME + PAD,
                    pattern=PATTERN)

# Sort stream by distance to volcano
stream.traces.sort(key=lambda tr: gps2dist_azimuth(VOLC_COORDS[0], VOLC_COORDS[1], tr.stats.latitude, tr.stats.longitude)[0])

# Attach response information
inventory = read_inventory(METADATA_FILEPATH)
stream.attach_response(inventory)

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
                                     dr_kwargs=DR_KWARGS,
                                     export_path=EXPORT_PATH,
                                     transparent=TRANSPARENT)



