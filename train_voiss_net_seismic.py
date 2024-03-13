# Import all dependencies
import tensorflow as tf
from toolbox import create_labeled_dataset, set_universal_seed, augment_labeled_dataset, train_voiss_net

# Disable GPU if desired
DISABLE_GPU = True

# Define inputs and parameters for create_labeled_dataset
json_filepath = './labels/voissnet_labels_seismic.json'
output_dir = './labeled_npy_files/seismic/'
label_dict = {'Broadband Tremor': 0,
              'Harmonic Tremor': 1,
              'Monochromatic Tremor': 2,
              'Earthquake': 3,
              'Explosion': 4,
              'Noise': 5}
transient_indices = [3, 4]  # indices of transient classes
time_step = 4 * 60  # s
source = 'IRIS'
network = 'AV'
station = 'PN7A,PS1A,PS4A,PV6A,PVV'
channel = '*HZ'
location = ''
pad = 240  # s
window_duration = 10  # s
freq_lims = (0.5, 10)  # Hz

# Create labeled dataset from json file and store in output directory
create_labeled_dataset(json_filepath, output_dir, label_dict, transient_indices, time_step, source, network, station,
                       location, channel, pad, window_duration, freq_lims)

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
set_universal_seed(19)

# Define inputs and parameters for augment_labeled_dataset
npy_dir = output_dir
omit_index = [0,3]  # do not include broadband tremor and earthquakes in count determination
noise_index = 5  # use noise samples to augment
testval_ratio = 0.2  # use 20% of sparse-est class count to pull test and validation sets
noise_ratio = 0.35  # weight of noise sample added for augmentation

# Augment labeled dataset and do training, validation and test set split
train_paths, valid_paths, test_paths = augment_labeled_dataset(npy_dir=npy_dir, omit_index=omit_index,
                                                               noise_index=noise_index,testval_ratio=testval_ratio,
                                                               noise_ratio=noise_ratio)

# Define inputs and parameters for train_voiss_net
model_tag = 'voissnet_4min_seismic_NEWTEST'
batch_size = 100  # default
learning_rate = 0.0005  # default
patience = 20  # epochs

# Train VOISS-Net model
train_voiss_net(train_paths=train_paths, valid_paths=valid_paths, test_paths=test_paths, label_dict=label_dict,
                model_tag=model_tag, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
