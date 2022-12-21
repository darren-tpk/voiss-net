# %% PLOT SPECTROGRAM(S) BULK

# Import all dependencies
import numpy as np
import glob
import matplotlib.pyplot as plt
from keras import layers, models, losses
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random
from DataGenerator import DataGenerator
from sklearn import metrics

# Define npy directory and filepath
spec_dir = '/Users/darrentpk/Desktop/PVV_npy/'
spec_paths = glob.glob(spec_dir + '*.npy')

# Shuffle the files
random.seed(930)
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
# Only initialize the test generator after the model is trained
train_gen = DataGenerator(train_paths, train_label_dict, **params)
valid_gen = DataGenerator(valid_paths, valid_label_dict, **params)

# Define the CNN

# Add singular channel to conform with tensorflow input dimensions
input_shape = [*eg_spec.shape, 1]

# Build a sequential model
model = models.Sequential()
# Convolutional layer, 32 filters, 3x3 kernel, 1x1 stride, padded to retain shape
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape, padding="same"))
# Max pooling layer, 2x2 pool, 2x2 stride
model.add(layers.MaxPooling2D((2, 2)))
# Convolutional layer, 64 filters, 3x3 kernel, 1x1 stride, padded to retain shape
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
# Max pooling layer, 2x2 pool, 2x2 stride
model.add(layers.MaxPooling2D((2, 2)))
# Convolutional layer, 128 filters, 3x3 kernel, 1x1 stride, padded to retain shape
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
# Max pooling layer, 2x2 pool, 2x2 stride
model.add(layers.MaxPooling2D((2, 2)))
# Flatten
model.add(layers.Flatten())
# Dense layer, 64 units
model.add(layers.Dense(64, activation="relu"))
# Dense layer, 6 units, one per class
model.add(layers.Dense(params["n_classes"], activation="softmax"))
# Compile model
model.compile(optimizer="adam", loss=losses.categorical_crossentropy, metrics=["accuracy"])
# Print out model summary
model.summary()

# Implement early stopping and checkpointing
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# fit model
history = model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=[es, mc])
# load the saved model
saved_model = load_model('best_model.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(trainX, trainy, verbose=0)
_, test_acc = saved_model.evaluate(validX, validy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

""" Fit the model to the data"""

history = model.fit_generator(generator=train_gen, validation_data=valid_gen, epochs=100, callbacks=[es,mc])

# make predictions on the validation data
valid_params = params.copy()
valid_params["batch_size"] = len(valid_labels)
valid_params["shuffle"] = False
valid_gen = DataGenerator(valid_paths, valid_label_dict, **valid_params)
valid = model.predict(valid_gen)

pred_labs = np.argmax(valid, axis=1)
true_labs = np.array(list(valid_gen.labels.values()))


""" Some Results """

# loss and accuracy plots
plt.ion()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(history.history["accuracy"], label="training")
axs[0].plot(history.history["val_accuracy"], label="testing")
axs[0].set_ylabel("accuracy")
axs[0].set_xlabel("epoch")

axs[1].plot(history.history["loss"], label="training")
axs[1].plot(history.history["val_loss"], label="testing")
axs[1].set_ylabel("loss")
axs[1].set_xlabel("epoch")

axs[0].legend()

# confusion matrix
confusion_matrix = metrics.confusion_matrix(true_labs, pred_labs)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)
plt.figure()
cm_display.plot()
