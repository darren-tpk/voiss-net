# Import all dependencies
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from toolbox import set_universal_seed, augment_labeled_dataset

#%% user input
VOLC = 'Pavlof'
SVDIR = '/Users/dfee/Documents/generalized_tremor/figures/'
SAVE = False
PLOT_AUGMENT = False

TESTVAL_RAT = 0.2


SPEC_DIR_PAVLOF = '/Users/dfee/Documents/generalized_tremor/labeled_npy_2min/'
label_dict_pavlof = {'Broadband Tremor': 0, 'Harmonic Tremor': 1,
                     'Monochromatic Tremor': 2, 'Non-tremor Signal': 3,
                     'Explosion': 4, 'Noise': 5}


SPEC_DIR_SEMI = '/Users/dfee/Documents/generalized_tremor/labeled_npy_2min_semi/'
label_dict_semi = {'Broadband Tremor': 0, 'Harmonic Tremor': 1,
                     'Monochromatic Tremor': 2, 'Non-tremor Signal': 3,
                     'Long Period': 4, 'Explosion': 5, 'Noise': 6}

rcParams['font.size'] = 8

#%% read in files and count classes

# Pavlof
result = label_dict_pavlof.items()
# Convert object to a list
data = list(result)
# Convert list to an array
numpyArray = np.array(data)
# Count the number of samples of each class
nclasses_pavlof = len(
    np.unique([filepath[-5] for filepath in glob.glob(SPEC_DIR_PAVLOF + '*.npy')]))
class_paths_pavlof = [glob.glob(SPEC_DIR_PAVLOF + '*_' + str(c) + '.npy') for c in
               range(nclasses_pavlof)]
class_counts_pavlof = np.array([len(paths) for paths in class_paths_pavlof])
print('\nPavlof Initial class counts:')
print(''.join(['%d: %d\n' % (c, class_counts_pavlof[c]) for c in range(nclasses_pavlof)]))

# now add empty LP class to Pavlof
label_dict_pavlof = label_dict_semi
class_counts_pavlof = np.insert(class_counts_pavlof, 4, 0)
nclasses_pavlof = nclasses_pavlof + 1



# Semi
result = label_dict_semi.items()
# Convert object to a list
data = list(result)
# Convert list to an array
numpyArray = np.array(data)
# Count the number of samples of each class
nclasses_semi = len(
    np.unique([filepath[-5] for filepath in glob.glob(SPEC_DIR_SEMI + '*.npy')]))
class_paths_semi = [glob.glob(SPEC_DIR_SEMI + '*_' + str(c) + '.npy') for c in
               range(nclasses_semi)]
class_counts_semi = np.array([len(paths) for paths in class_paths_semi])

print('\nSemi Initial class counts:')
print(''.join(['%d: %d\n' % (c, class_counts_semi[c]) for c in range(nclasses_semi)]))


#%% now plot

x_labels = []
for item in label_dict_semi.keys():
    if ' ' in item:
        split_items = item.split(' ')
        x_labels.append(f'{split_items[0]}\n{split_items[1]}')
    else:
        x_labels.append(item)

# plot stacked bar chart of class counts
class_counts_all = {
    "Pavlof": class_counts_pavlof,
    "Semi": class_counts_semi,
}
width = 0.75
bottom = np.zeros(nclasses_semi)
co = ['darkgray', 'darkred']


fig, ax = plt.subplots()
fig.set_size_inches(7, 5.25)
#plt.clf()
i=0
for boolean, class_count_tmp in class_counts_all.items():
    p = ax.bar(label_dict_semi.keys(), class_count_tmp, width, label=boolean,
               bottom=bottom, color=co[i])
    bottom += class_count_tmp
    i=i+1
ax.set_ylabel('Number of Samples')
ax.set_xticks(np.array(list(label_dict_semi.values())))
ax.set_xticklabels(x_labels)
ax.legend(class_counts_all.keys())


if SAVE:
    fig.savefig(f'{SVDIR}Pavlof-Semi_class_counts_2min.png', dpi=300, bbox_inches='tight')
#list(label_dict)[0].split()

#
# # plot historgram of class counts
# fig = plt.figure(1)
# fig.set_size_inches(9, 5.25)
# plt.clf()
# ax = fig.add_subplot(111)
# ax.bar(range(nclasses_pavlof), class_counts_pavlof, color='darkgray')
# ax.set_ylabel('Number of Samples')
# ax.set_xticks(np.array(list(label_dict_pavlof.values())))
# ax.set_xticklabels(label_dict_pavlof.keys(), rotation=45, ha='center')
#
# if SAVE:
#     fig.savefig(f'{SVDIR}Pavlof_2min_class_counts_fullt.png', dpi=300, bbox_inches='tight')
# #list(label_dict)[0].split()

#
# # Determine number of samples set aside for validation set and test set (each)
# testval_number = int(np.floor(np.min(class_counts) / (1 / TESTVAL_RAT)))
# print(
#     'Setting aside samples for validation and test set (%.1f%% of sparsest class count each)' % (
#                 TESTVAL_RAT * 100))
# print('%d samples kept for validation set (%d per class)' % (
# nclasses * testval_number, testval_number))
# print('%d samples kept for test set (%d per class)' % (
# nclasses * testval_number, testval_number))

# # Augmentation params
# omit_index = [0,3]  # do not include broadband tremor and non-tremor signal in count determination
# noise_index = 5  # use noise samples to augment
# testval_ratio = 0.2  # use 20% of sparse-est class count to pull test and validation sets
# noise_ratio = 0.35  # weight of noise sample added for augmentation
#
# # Configure train, validation and test paths and determine unique classes
# train_paths, valid_paths, test_paths = augment_labeled_dataset(npy_dir=SPEC_DIR, omit_index=omit_index,
#                                                              noise_index=noise_index,testval_ratio=testval_ratio,
#                                                              noise_ratio=noise_ratio)
# train_classes = [int(i.split("_")[-1][0]) for i in train_paths]
# unique_classes = np.unique(train_classes)
#
# # Count the number of samples of each augmented class
# train_counts = np.unique(train_classes, return_counts=True)[1]