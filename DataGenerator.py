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
    ):

        self.list_ids = list_ids
        self.labels = labels
        self.dim = dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_ids))

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

        # normalize the batch
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

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
