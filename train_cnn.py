import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import Sequential
import random
from DataGenerator import DataGenerator
from sklearn import metrics


""" Define where the data are """

project_dir = '/Users/darrentpk/Desktop/GitHub/tremor_ml/'
spec_dir = '/Users/darrentpk/Desktop/labeled_npy/'
spec_paths = glob.glob(spec_dir + '*.npy')

# shuffle the files
random.seed(930)
random.shuffle(spec_paths)

# grab class information from each file
classes = [int(i.split("_")[-1][0]) for i in spec_paths]
unique_classes = np.unique(classes)

# read in example file to get some information about the spectrograms
eg_spec = np.load(spec_paths[0])

# define parameters to generate the training, testing, validation data
params = {
    "dim": eg_spec.shape,
    "batch_size": 100,
    "n_classes": len(unique_classes),
    "shuffle": True,
}

""" Partition into training, testing, and validation data """

n_train_files = int(len(spec_paths) * 0.6)
n_test_files = int(len(spec_paths) * 0.2)
n_valid_files = int(len(spec_paths) * 0.2)

train_paths = spec_paths[0:n_train_files]
test_paths = spec_paths[n_train_files : (n_train_files + n_test_files)]
valid_paths = spec_paths[(n_train_files + n_test_files) :]

# create a dictionary holding labels for each path in the test and training data
train_classes = [int(i.split("_")[-1][0]) for i in train_paths]
train_labels = [np.where(i == unique_classes)[0][0] for i in train_classes]
train_label_dict = dict(zip(train_paths, train_labels))

test_classes = [int(i.split("_")[-1][0]) for i in test_paths]
test_labels = [np.where(i == unique_classes)[0][0] for i in test_classes]
test_label_dict = dict(zip(test_paths, test_labels))

valid_classes = [int(i.split("_")[-1][0]) for i in valid_paths]
valid_labels = [np.where(i == unique_classes)[0][0] for i in valid_classes]
valid_label_dict = dict(zip(valid_paths, valid_labels))

# instantiate the training and testing generators
# instantiate the validation generator after the model is trained
train_gen = DataGenerator(train_paths, train_label_dict, **params)
test_gen = DataGenerator(test_paths, test_label_dict, **params)


""" Define the CNN """

# tensorlow likes images with shapes [length x width x channels]
input_shape = [*eg_spec.shape, 1]

model = models.Sequential()
model.add(
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=input_shape,
        padding="same",
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(params["n_classes"], activation="softmax"))

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=["accuracy"],
)

model.summary()


""" Fit the model to the data"""

history = model.fit_generator(generator=train_gen, validation_data=test_gen, epochs=100)

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
