# Import all dependencies
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, losses, optimizers
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from DataGenerator import DataGenerator
from sklearn import metrics
from toolbox import set_universal_seed, augment_labeled_dataset
import tensorflow as tf

DISABLE_GPU = True 

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

# Set universal seed
set_universal_seed(42)

# Define npy and repo directories
npy_dir = '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/'
repo_dir = '/Users/darrentpk/Desktop/Github/tremor_ml/'

# Augmentation params
omit_index = [0,3]  # do not include broadband tremor and non-tremor signal in count determination
noise_index = 5  # use noise samples to augment
testval_ratio = 0.2  # use 20% of sparse-est class count to pull test and validation sets
noise_ratio = 0.35  # weight of noise sample added for augmentation

# Configure train, validation and test paths and determine unique classes
train_paths, valid_paths, test_paths = augment_labeled_dataset(npy_dir=npy_dir, omit_index=omit_index,
                                                             noise_index=noise_index,testval_ratio=testval_ratio,
                                                             noise_ratio=noise_ratio)
train_classes = [int(i.split("_")[-1][0]) for i in train_paths]
unique_classes = np.unique(train_classes)

# Define model name
model_type = '4min_all_augmented_drop'
model_name = repo_dir + 'models/' + model_type + '_model.h5'
meanvar_name = repo_dir + 'models/' + model_type + '_meanvar.npy'
curve_name = repo_dir + 'figures/' + model_type + '_curve.png'
confusion_name = repo_dir + 'figures/' + model_type + '_confusion.png'

# Read in example file to determine spectrogram shape
eg_spec = np.load(train_paths[0])

# Define parameters to generate the training, testing, validation data
params = {
    "dim": eg_spec.shape,
    "n_classes": len(unique_classes),
    "shuffle": True,
    "running_x_mean": np.mean(eg_spec),
    "running_x_var": np.var(eg_spec)
}

# Configure test and validation lists
valid_classes = [int(i.split("_")[-1][0]) for i in valid_paths]
test_classes = [int(i.split("_")[-1][0]) for i in test_paths]

# Create a dictionary holding labels for all filepaths in the training, testing, and validation data
train_labels = [np.where(i == unique_classes)[0][0] for i in train_classes]
train_label_dict = dict(zip(train_paths, train_labels))
valid_labels = [np.where(i == unique_classes)[0][0] for i in valid_classes]
valid_label_dict = dict(zip(valid_paths, valid_labels))
test_labels = [np.where(i == unique_classes)[0][0] for i in test_classes]
test_label_dict = dict(zip(test_paths, test_labels))

# Initialize the training and validation generators
train_gen = DataGenerator(train_paths, train_label_dict, batch_size=100, **params, is_training=True)
valid_gen = DataGenerator(valid_paths, valid_label_dict, batch_size=len(valid_paths), **params, is_training=False)

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
# Flatten and add 20% dropout to inputs
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
# Dense layer, 128 units with 50% dropout
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.5))
# Dense layer, 64 units with 50% dropout
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.5))
# Dense layer, 6 units, one per class
model.add(layers.Dense(params["n_classes"], activation="softmax"))
# Compile model
optimizer = optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=["accuracy"])
# Print out model summary
model.summary()
# Implement early stopping, checkpointing, and transference of mean and variance
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
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
axs[0].axvline(len(history.history["val_accuracy"])-es.patience-1,color='k',linestyle='--',alpha=0.5,label='early stop')
axs[0].set_ylabel("accuracy")
axs[0].set_xlabel("epoch")
axs[1].plot(history.history["loss"], label="training")
axs[1].plot(history.history["val_loss"], label="validation")
axs[1].axvline(len(history.history["val_loss"])-es.patience-1,color='k',linestyle='--',alpha=0.5)
axs[1].set_ylabel("loss")
axs[1].set_xlabel("epoch")
axs[0].legend()
fig.suptitle(model_type, fontweight='bold')
fig.savefig(curve_name)
fig.show()

# Create data generator for test data
test_params = {
    "dim": eg_spec.shape,
    "batch_size": len(test_labels),
    "n_classes": len(unique_classes),
    "shuffle": False}
test_gen = DataGenerator(test_paths, test_label_dict, **test_params, is_training=False,
                         running_x_mean=train_gen.running_x_mean, running_x_var=train_gen.running_x_var)

# Use saved model to make predictions
saved_model = load_model(model_name)
test = saved_model.predict(test_gen)
pred_labs = np.argmax(test, axis=1)
true_labs = np.array([test_gen.labels[id] for id in test_gen.list_ids])

# Print evaluation on test data
acc = metrics.accuracy_score(true_labs, pred_labs)
pre, rec, f1, _ = metrics.precision_recall_fscore_support(true_labs, pred_labs, average='macro')
metrics_chunk = model_name + '\n' + ('Accuracy: %.3f' % acc) + '\n' + ('Precision: %.3f' % pre) + '\n' + ('Recall: %.3f' % rec) + '\n' + ('F1 Score: %.3f' % f1)
print(metrics_chunk)
# with open("output.txt", "a") as outfile:
#     outfile.write(metrics_chunk + '\n')

# Confusion matrix
confusion_matrix = metrics.confusion_matrix(true_labs, pred_labs)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)
plt.figure()
cm_display.plot()
plt.title(model_type + '\nacc:%.3f, pre:%.3f, rec:%.3f, f1:%.3f' % (acc,pre,rec,f1),fontweight='bold')
plt.savefig(confusion_name)
plt.show()

# # Now conduct post-mortem
# path_pred_true = np.transpose([test_paths, pred_labs, true_labs])
# label_dict = {0: 'Broadband Tremor',
#               1: 'Harmonic Tremor',
#               2: 'Monochromatic Tremor',
#               3: 'Non-tremor Signal',
#               4: 'Explosion',
#               5: 'Noise'}
#
# variety = ['0','1','2','3','4','5']
#
# for predicted_label in variety:
#     for true_label in variety:
#         N = 9
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