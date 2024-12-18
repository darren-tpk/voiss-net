import numpy as np
import keras
import math


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        list_ids,
        labels,
        dim,
        n_classes,
        batch_size=100,
        shuffle=True,
        is_training=True,
        running_x_mean=None,
        running_x_var=None,
    ):

        self.list_ids = list_ids
        self.labels = labels
        self.dim = dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_ids))
        self.is_training = is_training
        self.running_x_mean = running_x_mean
        self.running_x_var = running_x_var

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples
        You could do more customizations here. For example,
        alternate normalizations, data augmentation, etc..."""

        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # generate data
        for i, id in enumerate(list_ids_temp):

            # store sample
            x[i, :] = np.load(id)

            # store label
            y[i] = self.labels[id]

        # Standardize data if needed
        # If training with running mean and variance, standardize and update running mean and variance
        if self.is_training and self.running_x_mean is not None and self.running_x_var is not None:

            x_mean = np.mean(x, axis=0)
            x_var = np.var(x, axis=0)

            x = (x - x_mean) / np.sqrt(x_var + 1e-5)

            self.running_x_mean = 0.9 * self.running_x_mean + 0.1 * x_mean
            self.running_x_var = 0.9 * self.running_x_var + 0.1 * x_var

        # If generating validation and test data generators, standardize with running mean and variance
        elif not self.is_training and self.running_x_mean is not None and self.running_x_var is not None:
            x = (x - self.running_x_mean) / np.sqrt(self.running_x_var + 1e-5)

        # If not using running mean and variance, simply pass the data through
        else:
            pass

        # Min-max scale by individual image
        for k in range(self.batch_size):
            x[k, :] = (x[k, :] - np.min(x[k, :])) / (np.max(x[k, :]) - np.min(x[k, :]))

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getitem__(self, index):
        """generate one batch of data"""
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # find list of ids
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def __len__(self):
        """ Define the number of batches per epoch"""
        return math.ceil(len(self.list_ids) / self.batch_size)

