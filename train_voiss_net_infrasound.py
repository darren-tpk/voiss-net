# Import all dependencies
import tensorflow as tf
from toolbox import create_labeled_dataset, set_universal_seed, augment_labeled_dataset, train_voiss_net

# Disable GPU for model training if desired
DISABLE_GPU = True

# Define inputs and parameters for training the model
NPY_DIR = './labeled_npy_files/infrasound/'
MODEL_TAG = 'voissnet_infrasound'
BATCH_SIZE = 100
LEARNING_RATE = 0.0005
PATIENCE = 20  # epochs

# Define inputs and parameters for augment_labeled_dataset
OMIT_INDEX = [0, 3]  # do not include in class count determination
NOISE_INDEX = 2  # use noise samples to augment
TESTVAL_RATIO = 0.2  # use this ratio of sparse-est class to pull test and validation sets
NOISE_RATIO = 0.35  # weight of noise sample added for augmentation

LABEL_DICT = {'Infrasonic Tremor': 0,
              'Explosion': 1,
              'Wind Noise': 2,
              'Electronic Noise': 3}

# If npy labels are already created, then leave this as False. Otherwise,
# provide json file and relevant parameters
CREATE_LABELED_DATASET = False

if CREATE_LABELED_DATASET:
    JSON_FILEPATH = './labels/voissnet_labels_infrasound.json'
    TRANSIENT_INDICES = [1]  # indices of transient classes
    TIME_STEP = 4 * 60  # s
    SOURCE = 'IRIS'
    NETWORK = 'AV'
    STATION = 'PN7A,PS1A,PS4A,PV6A,PVV'
    CHANNEL = '*DF'
    LOCATION = ''
    PAD = 240  # s
    WINDOW_DURATION = 10  # s
    FREQ_LIMS = (0.5, 10)  # Hz

    # Create labeled dataset from json file and store in output directory
    create_labeled_dataset(JSON_FILEPATH, NPY_DIR, LABEL_DICT,
                           TRANSIENT_INDICES, TIME_STEP, SOURCE, NETWORK,
                           STATION, LOCATION, CHANNEL, PAD, WINDOW_DURATION,
                           FREQ_LIMS)


# Disable GPU if desired
if DISABLE_GPU:
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
else:
    tf.keras.backend.set_floatx('float32')

# Set universal seed for dataset augmentation and model training
set_universal_seed(43)

# Augment labeled dataset and do training, validation and test set split
train_paths, valid_paths, test_paths = augment_labeled_dataset(npy_dir=NPY_DIR,
                                                               omit_index=OMIT_INDEX,
                                                               noise_index=NOISE_INDEX,
                                                               testval_ratio=TESTVAL_RATIO,
                                                               noise_ratio=NOISE_RATIO)

# Train VOISS-Net model
train_voiss_net(train_paths=train_paths, valid_paths=valid_paths,
                test_paths=test_paths, label_dict=LABEL_DICT,
                model_tag=MODEL_TAG, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, patience=PATIENCE,
                meanvar_standardization=True)
