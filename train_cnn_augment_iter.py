# Import all dependencies
import numpy as np
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

# Define npy and repo directories
seis_npy_dir = '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/'
infra_npy_dir = '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min_infra/'
repo_dir = '/Users/darrentpk/Desktop/Github/tremor_ml/'

# Open text file to store metrics
seis_metrics = open('./seis_metrics.txt', 'w')
seis_metrics.write('SD, ACCURACY, PRECISION, RECALL, F1SCORE')
infra_metrics = open('./infra_metrics.txt', 'w')
infra_metrics.write('SD, ACCURACY, PRECISION, RECALL, F1SCORE')

# Set universal seed
for seed in range(1,51):
    set_universal_seed(seed)

    # Augmentation params
    omit_index = [0,3]  # do not include broadband tremor and non-tremor signal in count determination
    noise_index = 5  # use noise samples to augment
    testval_ratio = 0.2  # use 20% of sparse-est class count to pull test and validation sets
    noise_ratio = 0.35  # weight of noise sample added for augmentation

    # Configure train, validation and test paths and determine unique classes
    train_paths, valid_paths, test_paths = augment_labeled_dataset(npy_dir=seis_npy_dir, omit_index=omit_index,
                                                                 noise_index=noise_index,testval_ratio=testval_ratio,
                                                                 noise_ratio=noise_ratio)
    train_classes = [int(i.split("_")[-1][0]) for i in train_paths]
    unique_classes = np.unique(train_classes)

    # Define model name
    model_type = '4min_all_augmented_iter'
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
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    emv = ExtractMeanVar()
    # Fit model
    history = model.fit(train_gen, validation_data=valid_gen, epochs=200, callbacks=[es, mc, emv])

    # Save the final running mean and variance
    np.save(meanvar_name, [train_gen.running_x_mean,train_gen.running_x_var])

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
    seis_metrics.write('\n%d, %.5f, %.5f, %.5f, %.5f' % (seed, acc, pre, rec, f1))

# Close text file
seis_metrics.close()

# Set universal seed
for seed in range(1,51):
    set_universal_seed(seed)

    # Augmentation params
    omit_index = [0,3]  # do not include electronic noise in count determination
    noise_index = 2  # use noise samples to augment
    testval_ratio = 0.2  # use 20% of sparse-est class count to pull test and validation sets
    noise_ratio = 0.35  # weight of noise sample added for augmentation

    # Configure train, validation and test paths and determine unique classes
    train_paths, valid_paths, test_paths = augment_labeled_dataset(npy_dir=infra_npy_dir, omit_index=omit_index,
                                                                 noise_index=noise_index,testval_ratio=testval_ratio,
                                                                 noise_ratio=noise_ratio)
    train_classes = [int(i.split("_")[-1][0]) for i in train_paths]
    unique_classes = np.unique(train_classes)

    # Define model name
    model_type = '4min_all_augmented_infra_iter'
    model_name = repo_dir + '/models/' + model_type + '_model.h5'
    meanvar_name = repo_dir + '/models/' + model_type + '_meanvar.npy'
    curve_name = repo_dir + '/figures/' + model_type + '_curve.png'
    confusion_name = repo_dir + '/figures/' + model_type + '_confusion.png'

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
    infra_metrics.write('\n%d, %.5f, %.5f, %.5f, %.5f' % (seed, acc, pre, rec, f1))

# Close text file
infra_metrics.close()