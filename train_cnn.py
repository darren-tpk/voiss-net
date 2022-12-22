# %% PLOT SPECTROGRAM(S) BULK

# Import all dependencies
import numpy as np
import glob
import matplotlib.pyplot as plt
from keras import layers, models, losses
from keras.models import load_model
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
# Fit model
history = model.fit(train_gen, validation_data=valid_gen, epochs=100, callbacks=[es, mc])

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

# Create data generator for test data
test_params = params.copy()
test_params["batch_size"] = len(test_labels)
test_params["shuffle"] = False
test_gen = DataGenerator(test_paths, test_label_dict, **test_params)

# Use saved model to make predictions
saved_model = load_model('best_model.h5')
test = saved_model.predict(test_gen)
pred_labs = np.argmax(test, axis=1)
true_labs = np.array(list(test_gen.labels.values()))

# Confusion matrix
confusion_matrix = metrics.confusion_matrix(true_labs, pred_labs)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)
plt.figure()
cm_display.plot()

# Print evaluation on test data
acc = metrics.accuracy_score(true_labs, pred_labs)
pre, rec, f1, _ = metrics.precision_recall_fscore_support(true_labs, pred_labs, average='weighted')
print('Accuracy: %.3f' % acc)
print('Precision: %.3f' % pre)
print('Recall: %.3f' % rec)
print('F1 Score: %.3f' % f1)

# Now conduct post-mortem
path_pred_true = np.transpose([test_paths, pred_labs, true_labs])
label_dict = {0: 'Broadband Tremor',
              1: 'Harmonic Tremor',
              2: 'Monochromatic Tremor',
              3: 'Non-tremor Signal',
              4: 'Explosion',
              5: 'Noise'}

variety = ['3','4']

for predicted_label in variety:
    for true_label in variety:
        corresponding_filenames = [p[0] for p in path_pred_true if p[1]==predicted_label and p[2]==true_label]
        N = 16
        import colorcet as cc
        fig, axs = plt.subplots(nrows=int(np.sqrt(N)), ncols=int(np.sqrt(N)), figsize=(7, 10))
        fig.suptitle('%s predicted as %s (total = %d)' % (label_dict[int(predicted_label)], label_dict[int(true_label)], len(corresponding_filenames)))
        for i in range(int(np.sqrt(N))):
            for j in range(int(np.sqrt(N))):
                filename_index = i * int(np.sqrt(N)) + (j + 1) - 1
                if filename_index > (len(corresponding_filenames) - 1):
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    continue
                else:
                    spec_db = np.load(corresponding_filenames[filename_index])
                    if np.sum(spec_db < -250) > 0:
                        print(i, j)
                    axs[i, j].imshow(spec_db, vmin=np.percentile(spec_db, 20), vmax=np.percentile(spec_db, 97.5),
                                     origin='lower', aspect='auto', interpolation=None, cmap=cc.cm.rainbow)
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
        fig.show()