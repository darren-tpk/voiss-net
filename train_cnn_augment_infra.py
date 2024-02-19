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
set_universal_seed(43)

# Get list of spectrogram slice paths based on station option
npy_dir = '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min_infra/'
repo_dir = '/Users/darrentpk/Desktop/GitHub/tremor_ml/'

# Augmentation params
omit_index = [0,3]  # do not include tremor and electronic noise in count determination
noise_index = 2  # use noise samples to augment
testval_ratio = 0.2  # use 20% of sparse-est class count to pull test and validation sets
noise_ratio = 0.35  # weight of noise sample added for augmentation

# Configure train, validation and test paths and determine unique classes
train_paths, valid_paths, test_paths = augment_labeled_dataset(npy_dir=npy_dir, omit_index=omit_index,
                                                             noise_index=noise_index,testval_ratio=testval_ratio,
                                                             noise_ratio=noise_ratio)
train_classes = [int(i.split("_")[-1][0]) for i in train_paths]
unique_classes = np.unique(train_classes)

# Define model name
MODEL_TYPE = '4min_all_augmented_infra'
model_name = repo_dir + 'models/' + MODEL_TYPE + '_model.h5'
meanvar_name = repo_dir + 'models/' + MODEL_TYPE + '_meanvar.npy'
history_name = repo_dir + 'models/' + MODEL_TYPE + '_history.npy'
curve_name = repo_dir + 'figures/' + MODEL_TYPE + '_curve.png'
confusion_name = repo_dir + 'figures/' + MODEL_TYPE + '_confusion.png'

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
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
emv = ExtractMeanVar()
# Fit model
history = model.fit(train_gen, validation_data=valid_gen, epochs=200, callbacks=[es, mc, emv])

# Save the final running mean and variance
np.save(meanvar_name, [train_gen.running_x_mean,train_gen.running_x_var])

# Save model training history to reproduce learning curves
np.save(history_name, history.history)

# Plot loss and accuracy curves
fig_curves, axs = plt.subplots(1, 2, figsize=(8, 5))
axs[0].plot(history.history["accuracy"], label="Training", lw=1)
axs[0].plot(history.history["val_accuracy"], label="Validation", lw=1)
axs[0].axvline(len(history.history["val_accuracy"])-es.patience-1,color='k', linestyle='--',alpha=0.5,label='Early Stop')
axs[0].set_ylim([0,1])
axs[0].set_ylabel("Accuracy")
axs[0].set_xlabel("Epoch")
axs[1].plot(history.history["loss"], label="Training", lw=1)
axs[1].plot(history.history["val_loss"], label="Validation", lw=1)
axs[1].axvline(len(history.history["val_loss"])-es.patience-1,color='k', linestyle='--',alpha=0.5,label='Early Stop')
axs[1].set_ylabel("Loss")
axs[1].set_xlabel("Epoch")
axs[0].legend()
fig_curves.suptitle(MODEL_TYPE, fontweight='bold')
fig_curves.show()

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

# Confusion matrix
import colorcet as cc
class_labels = ['Infrasonic\nTremor','Explosion','Wind\nNoise','Electronic\nNoise']
confusion_matrix = metrics.confusion_matrix(true_labs, pred_labs, normalize='true')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=class_labels)
cm_display.plot(xticks_rotation=30,cmap=cc.cm.blues, values_format='.2f', im_kw=dict(vmin=0, vmax=1))
fig_cm=plt.gcf()
fig_cm.set_size_inches(8, 6.25)
plt.title(MODEL_TYPE + '\nAccuracy: %.3f, Precision :%.3f,\nRecall:%.3f, F1 Score:%.3f' % (acc,pre,rec,f1),fontweight='bold')
fig_cm.savefig(confusion_name, bbox_inches='tight')
fig_cm.show()