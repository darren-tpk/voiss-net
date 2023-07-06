# Import all dependencies
import numpy as np
import glob
import matplotlib.pyplot as plt
from keras import layers, models, losses
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import random
from DataGenerator import DataGenerator
from sklearn import metrics
from itertools import compress

# Choose to balance training set or pre-merge any classes
balance_training = True
balance_type = 'subsampled2'  # 'oversampled', 'undersampled', or 'subsampled[0,1,2,3,4,5]'
random.seed(930)

# Train station-generic model
station_option = 'all'

# Get list of spectrogram slice paths based on station option
spec_dir = '/Users/dfee/Documents/generalized_tremor/labeled_npy_4min/'
repo_dir =  '/Users/dfee/repos/tremor_ml/'

if station_option == 'all':
    spec_paths = glob.glob(spec_dir + '*.npy')
else:
    spec_paths = glob.glob(spec_dir + station_option + '*.npy')

# Balance the training set if desired
if balance_training:
    # Get unique classes
    train_class_keys = [path.split('_')[-1][0] for path in spec_paths]
    unique_train_class_keys = list(set(train_class_keys))
    # Balance dataset using desired strategy
    if balance_type == 'undersampled':
        # Count unique classes and establish minimum class
        train_class_keys_count = [train_class_keys.count(class_key) for class_key in unique_train_class_keys]
        train_class_count_min = np.min(train_class_keys_count)
        # Loop through unique classes to randomly sample n_min file paths
        spec_paths_samples = []
        for class_key in unique_train_class_keys:
            subset_bool = [train_class_key==class_key for train_class_key in train_class_keys]
            spec_paths_subset = list(compress(spec_paths, subset_bool))
            spec_paths_sample = random.sample(spec_paths_subset,train_class_count_min)
            spec_paths_samples = spec_paths_samples + spec_paths_sample
        # Replace file path list
        spec_paths = spec_paths_samples
    elif balance_type == 'oversampled':
        # Count unique classes and establish maximum class
        train_class_keys_count = [train_class_keys.count(class_key) for class_key in unique_train_class_keys]
        train_class_count_max = np.max(train_class_keys_count)
        # Loop over unique classes
        spec_paths_oversamples = []
        for i, unique_train_class_key in enumerate(unique_train_class_keys):
            count_difference = train_class_count_max - train_class_keys_count[i]
            corresponding_spec_path_list = list(compress(spec_paths,[key == unique_train_class_key for key in train_class_keys]))
            corresponding_oversamples = list(np.random.choice(corresponding_spec_path_list, count_difference))
            spec_paths_oversamples = spec_paths_oversamples + corresponding_oversamples
        # Replace file path list
        spec_paths = spec_paths + spec_paths_oversamples
    elif balance_type[:-1] == 'subsampled':
        # Count unique classes and establish maximum class
        train_class_keys_count = [train_class_keys.count(class_key) for class_key in unique_train_class_keys]
        train_class_count_desired = train_class_keys_count[unique_train_class_keys.index(balance_type[-1])]
        # Loop over unique classes
        spec_paths_subsamples = []
        for i, unique_train_class_key in enumerate(unique_train_class_keys):
            count_difference = train_class_count_desired - train_class_keys_count[i]
            corresponding_spec_path_list = list(
                compress(spec_paths, [key == unique_train_class_key for key in train_class_keys]))
            if count_difference == 0:
                spec_paths_subsamples = spec_paths_subsamples + corresponding_spec_path_list
            elif count_difference > 0:
                corresponding_oversamples = list(np.random.choice(corresponding_spec_path_list, count_difference))
                spec_paths_subsamples = spec_paths_subsamples + corresponding_spec_path_list + corresponding_oversamples
            else:
                corresponding_undersamples = list(np.random.choice(corresponding_spec_path_list, train_class_count_desired))
                spec_paths_subsamples = spec_paths_subsamples + corresponding_undersamples
        # Replace file path list
        spec_paths = spec_paths_subsamples

# Define model name
model_type = '4min_' + station_option
if balance_training:
    model_type = model_type + '_' + balance_type
model_name = repo_dir + '/models/' + model_type + '_model.h5'
meanvar_name = repo_dir + '/models/' + model_type + '_meanvar.npy'
curve_name = repo_dir + '/figures/' + model_type + '_curve.png'
confusion_name = repo_dir + '/figures/' + model_type + '_confusion.png'

# Shuffle the files
random.shuffle(spec_paths)

# Pull class information from each file
classes = [int(i.split("_")[-1][0]) for i in spec_paths]
unique_classes = np.unique(classes)

# Read in example file to determine spectrogram shape
eg_spec = np.load(spec_paths[0])

# Define parameters to generate the training, testing, validation data
params = {
    "dim": eg_spec.shape,
    "batch_size": 100,
    "n_classes": len(unique_classes),
    "shuffle": True,
}

# Partition into training, testing, and validation data
n_train_files = int(len(spec_paths) * 0.6)
n_valid_files = int(len(spec_paths) * 0.2)
n_test_files = int(len(spec_paths) * 0.2)
train_paths = spec_paths[0:n_train_files]
valid_paths = spec_paths[n_train_files:(n_train_files + n_valid_files)]
test_paths = spec_paths[(n_train_files + n_valid_files):]

# Create a dictionary holding labels for all filepaths in the training, testing, and validation data
train_classes = [int(i.split("_")[-1][0]) for i in train_paths]
train_labels = [np.where(i == unique_classes)[0][0] for i in train_classes]
train_label_dict = dict(zip(train_paths, train_labels))

valid_classes = [int(i.split("_")[-1][0]) for i in valid_paths]
valid_labels = [np.where(i == unique_classes)[0][0] for i in valid_classes]
valid_label_dict = dict(zip(valid_paths, valid_labels))

test_classes = [int(i.split("_")[-1][0]) for i in test_paths]
test_labels = [np.where(i == unique_classes)[0][0] for i in test_classes]
test_label_dict = dict(zip(test_paths, test_labels))

# Initialize the training and validation generators
train_gen = DataGenerator(train_paths, train_label_dict, **params, is_training=True)
valid_gen = DataGenerator(valid_paths, valid_label_dict, **params, is_training=False)

# Define a Callback class that allows the validation dataset to adopt the running
# training mean and variance for normalization
class ExtractMeanVar(Callback):
    def on_epoch_end(self, epoch, logs=None):
        valid_gen.running_x_mean = train_gen.running_x_mean
        valid_gen.running_x_var = train_gen.running_x_var

# Define the CNN

# Add singular channel to conform with tensorflow input dimensions
input_shape = [*eg_spec.shape, 1]

# Build a sequential model
model = models.Sequential()
# Convolutional layer, 32 filters, 3x3 kernel, 1x1 stride, padded to retain shape
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape, padding="same"))
# Max pooling layer, 3x3 pool, 3x3 stride
model.add(layers.MaxPooling2D((3, 3)))
# Convolutional layer, 64 filters, 3x3 kernel, 1x1 stride, padded to retain shape
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
# Max pooling layer, 3x3 pool, 3x3 stride
model.add(layers.MaxPooling2D((3, 3)))
# Convolutional layer, 128 filters, 3x3 kernel, 1x1 stride, padded to retain shape
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
# Max pooling layer, 3x3 pool, 3x3 stride
model.add(layers.MaxPooling2D((3, 3)))
# Flatten
model.add(layers.Flatten())
# Dense layer, 128 units
model.add(layers.Dense(128, activation="relu"))
# Dense layer, 64 units
model.add(layers.Dense(64, activation="relu"))
# Dense layer, 6 units, one per class
model.add(layers.Dense(params["n_classes"], activation="softmax"))
# Compile model
model.compile(optimizer="adam", loss=losses.categorical_crossentropy, metrics=["accuracy"])
# Print out model summary
model.summary()
# Implement early stopping, checkpointing, and transference of mean and variance
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint(model_name, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
emv = ExtractMeanVar()
# Fit model
history = model.fit(train_gen, validation_data=valid_gen, epochs=200, callbacks=[es, mc, emv])

# Save the final running mean and variance
np.save(meanvar_name, [train_gen.running_x_mean,train_gen.running_x_var])

# Plot loss and accuracy curves
plt.ion()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(history.history["accuracy"], label="training")
axs[0].plot(history.history["val_accuracy"], label="validation")
axs[0].set_ylabel("accuracy")
axs[0].set_xlabel("epoch")
axs[1].plot(history.history["loss"], label="training")
axs[1].plot(history.history["val_loss"], label="validation")
axs[1].set_ylabel("loss")
axs[1].set_xlabel("epoch")
axs[0].legend()
fig.suptitle(model_type, fontweight='bold')
fig.savefig(curve_name)
fig.show()

# Create data generator for test data
test_params = params.copy()
test_params["batch_size"] = len(test_labels)
test_params["shuffle"] = False
test_gen = DataGenerator(test_paths, test_label_dict, **test_params, is_training=False,
                         running_x_mean=train_gen.running_x_mean, running_x_var=train_gen.running_x_var)

# Use saved model to make predictions
saved_model = load_model(model_name)
test = saved_model.predict(test_gen)
pred_labs = np.argmax(test, axis=1)
true_labs = np.array([test_gen.labels[id] for id in test_gen.list_ids])

# Confusion matrix
confusion_matrix = metrics.confusion_matrix(true_labs, pred_labs)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)
plt.figure()
cm_display.plot()
plt.title(model_type,fontweight='bold')
plt.savefig(confusion_name)
plt.show()

# Print evaluation on test data
acc = metrics.accuracy_score(true_labs, pred_labs)
pre, rec, f1, _ = metrics.precision_recall_fscore_support(true_labs, pred_labs, average='weighted')
metrics_chunk = model_name + '\n' + ('Accuracy: %.3f' % acc) + '\n' + ('Precision: %.3f' % pre) + '\n' + ('Recall: %.3f' % rec) + '\n' + ('F1 Score: %.3f' % f1)
print(metrics_chunk)
with open("output.txt", "a") as outfile:
    outfile.write(metrics_chunk + '\n')

# # Now conduct post-mortem
# path_pred_true = np.transpose([test_paths, pred_labs, true_labs])
# label_dict = {0: 'Broadband Tremor',
#               1: 'Harmonic Tremor',
#               2: 'Monochromatic Tremor',
#               3: 'Non-tremor Signal',
#               4: 'Explosion',
#               5: 'Noise'}
#
# variety = ['5']
#
# for predicted_label in variety:
#     for true_label in variety:
#         N = 16
#         corresponding_filenames = [p[0] for p in path_pred_true if p[1]==predicted_label and p[2]==true_label]
#         corresponding_filenames_chosen = random.sample(corresponding_filenames, N)
#         import colorcet as cc
#         fig, axs = plt.subplots(nrows=int(np.sqrt(N)), ncols=int(np.sqrt(N)), figsize=(7, 10))
#         fig.suptitle('%s predicted as %s (total = %d)' % (label_dict[int(true_label)], label_dict[int(predicted_label)], len(corresponding_filenames)))
#         for i in range(int(np.sqrt(N))):
#             for j in range(int(np.sqrt(N))):
#                 filename_index = i * int(np.sqrt(N)) + (j + 1) - 1
#                 if filename_index > (len(corresponding_filenames_chosen) - 1):
#                     axs[i, j].set_xticks([])
#                     axs[i, j].set_yticks([])
#                     continue
#                 else:
#                     spec_db = np.load(corresponding_filenames_chosen[filename_index])
#                     if np.sum(spec_db < -250) > 0:
#                         print(i, j)
#                     axs[i, j].imshow(spec_db, vmin=np.percentile(spec_db, 20), vmax=np.percentile(spec_db, 97.5),
#                                      origin='lower', aspect='auto', interpolation=None, cmap=cc.cm.rainbow)
#                     axs[i, j].set_xticks([])
#                     axs[i, j].set_yticks([])
#         fig.show()