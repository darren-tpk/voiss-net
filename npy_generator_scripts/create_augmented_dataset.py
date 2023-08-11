# Import all dependencies
import os
import glob
import shutil
import random
import numpy as np

# Define original labeled npy dir and augmented npy dir
npy_dir = '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/'
augmented_dir = '/Users/darrentpk/Desktop/all_npys/augmented_npy_4min/'
omit_index = 3  # do not include non-tremor signal in augmented count determination
noise_index = 5  # use noise samples to augment oversamples
noise_ratio = 0.35

# Count the number of samples of each class and calculate augmented number
nclasses = len(np.unique([filepath[-5] for filepath in glob.glob(npy_dir + '*.npy')]))
class_paths = [glob.glob(npy_dir + '*_' + str(c) + '.npy') for c in range(nclasses)]
class_counts = np.array([len(paths) for paths in class_paths])
testval_number = int(np.floor(np.min(class_counts)/5))  # ~20% of samples used in test, 20% in validation
leftover_counts = class_counts - 2*testval_number
augmented_number = int(np.floor(np.sum(leftover_counts[leftover_counts != leftover_counts[omit_index]])
                                / (nclasses-1)))

# Commence augmentation
print('Creating augmented directory and nested validation and test directories...')
try:
    os.mkdir(augmented_dir)
    os.mkdir(augmented_dir + 'test/')
    os.mkdir(augmented_dir + 'validation/')
    print('Augmented directory and nested directories created.\n')
except:
    print('Augmented directory already exists.\n')

# Loop over classes and copy over files that do not require augmentation
print('Copying over files that do not require augmentation...')
for c in range(nclasses):

    # Copy over test and validation subsets first
    test_list = list(np.random.choice(class_paths[c], testval_number, replace=False))
    leftover_list = [filepath for filepath in class_paths[c] if filepath not in test_list]
    val_list = list(np.random.choice(leftover_list, testval_number, replace=False))
    leftover_list = [filepath for filepath in leftover_list if filepath not in val_list]
    for class_path in test_list:
        shutil.copy(class_path, augmented_dir + 'test/' + class_path.split('/')[-1])
    for class_path in val_list:
        shutil.copy(class_path, augmented_dir + 'validation/' + class_path.split('/')[-1])

    # Handle leftover filepaths depending on index
    if c == omit_index:
        copy_list = list(np.random.choice(leftover_list, augmented_number, replace=False))
        for class_path in copy_list:
            shutil.copy(class_path, augmented_dir + class_path.split('/')[-1])
    elif c == noise_index:
        copy_list = list(np.random.choice(leftover_list, augmented_number, replace=False))
        noise_samples = [filepath for filepath in leftover_list if filepath not in copy_list]
        for class_path in copy_list:
            shutil.copy(class_path, augmented_dir + class_path.split('/')[-1])
    elif len(leftover_list) <= augmented_number:
        for class_path in leftover_list:
            shutil.copy(class_path, augmented_dir + class_path.split('/')[-1])
    else:
        raise ValueError('A class has more samples than the augment number. Check class counts!')
print('Copy complete.\n')

# Commence augmentation
print('Commencing augmentation...')

# Randomly sample based on count difference
augment_samples = []
augment_paths = [glob.glob(augmented_dir + '*_' + str(c) + '.npy') for c in range(nclasses)]
for c in range(nclasses):
    if c == omit_index or c == noise_index:
        continue
    else:
        count_difference = augmented_number - len(augment_paths[c])
        augment_samples = augment_samples + list(np.random.choice(augment_paths[c], count_difference, replace=True))

# Check if augment_samples and noise_samples have the same length
if len(augment_samples) == len(noise_samples):
    print('Augmentation samples and noise samples match in length.\n')
else:
    raise ValueError('Augmentation samples and noise samples do NOT in length.')

# Shuffle and add noise to augment samples
print('Shuffling and adding noise samples to augment samples...')
random.shuffle(augment_samples)
random.shuffle(noise_samples)
for augment_sample, noise_sample in zip(augment_samples,noise_samples):

    # Load both images and sum them using noise ratio
    augment_image = np.load(augment_sample)
    noise_image = np.load(noise_sample)
    augmented_image = ((1-noise_ratio) * augment_image) + (noise_ratio * noise_image)

    # Determine filepath by checking for uniqueness
    n = 0
    augmented_filepath = augmented_dir + augment_sample.split('/')[-1][:-4] + 'aug' + str(n) + '.npy'
    while os.path.isfile(augmented_filepath):
        n += 1
        augmented_filepath = augmented_dir + augment_sample.split('/')[-1][:-4] + 'aug' + str(n) + '.npy'
    np.save(augmented_filepath, augmented_image)
print('Done!')

# # Plot examples
# import matplotlib.pyplot as plt
# import colorcet as cc
# ns = np.random.choice(range(len(augment_samples)), 10)
# for n in ns:
#     augment_image = np.load(augment_samples[n])
#     noise_image = np.load(noise_samples[n])
#     augmented_image = ((1 - noise_ratio) * augment_image) + (noise_ratio * noise_image)
#     fig, ax = plt.subplots(1, 3, figsize=(8, 3))
#     ax[0].imshow(augment_image, vmin=np.percentile(augment_image, 20), vmax=np.percentile(augment_image, 97.5),
#                  origin='lower', aspect='auto', interpolation='None', cmap=cc.cm.rainbow)
#     ax[0].set_xticks([])
#     ax[0].set_yticks([])
#     ax[0].set_title('Class Sample')
#     ax[1].imshow(noise_image, vmin=np.percentile(noise_image, 20), vmax=np.percentile(noise_image, 97.5),
#                  origin='lower', aspect='auto', interpolation='None', cmap=cc.cm.rainbow)
#     ax[1].set_xticks([])
#     ax[1].set_yticks([])
#     ax[1].set_title('Noise Sample ' + str('%.2f' % noise_ratio))
#     ax[2].imshow(augmented_image, vmin=np.percentile(augmented_image, 20), vmax=np.percentile(augmented_image, 97.5),
#                  origin='lower', aspect='auto', interpolation='None', cmap=cc.cm.rainbow)
#     ax[2].set_xticks([])
#     ax[2].set_yticks([])
#     ax[2].set_title('Augmented Sample')
#     fig.show()