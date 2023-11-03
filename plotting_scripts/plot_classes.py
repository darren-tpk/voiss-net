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


SPEC_DIR = '/Users/dfee/Documents/generalized_tremor/labeled_npy_2min_all/'
label_dict= {'Broadband Tremor': 0, 'Harmonic Tremor': 1,
                     'Monochromatic Tremor': 2, 'Non-tremor Signal': 3,
                     'Long Period': 4, 'Explosion': 5, 'Noise': 6}

VOLC_STA_WC = {'Pavlof':'P*', 'Semisopochnoi':'C*'}
rcParams['font.size'] = 8

#%% read in files and count classes

result = label_dict.items()
# Convert object to a list
data = list(result)
# Convert list to an array
numpyArray = np.array(data)
# Count the number of samples of each class
nclasses = len(
    np.unique([filepath[-5] for filepath in glob.glob(SPEC_DIR + '*.npy')]))
class_paths = [glob.glob(SPEC_DIR + '*_' + str(c) + '.npy') for c in
               range(nclasses)]
class_counts = np.array([len(paths) for paths in class_paths])
print('\nInitial class counts:')
print(f'Total files: {np.sum(class_counts)}')
print(''.join(['%d: %d\n' % (c, class_counts[c]) for c in range(nclasses)]))

import fnmatch

VOLC_STA_WC = {'Pavlof':['PN7A', 'PVV', 'PV6A','PN7A','PS1A'],
               'Semisopochnoi':['CERB','CESW','CEPE','CERA','CETU','CEAP']}

class_counts_split = np.zeros((2,nclasses))
for i,key in enumerate(VOLC_STA_WC):
    #count1=[]
    #pattern = ['PN7A', 'PVV', 'PV6A','PN7A','PS1A']
    # loop over each class
    for j,paths in enumerate(class_paths):
        #count values for each station in each class and add them up
        count = 0
        for pat in VOLC_STA_WC[key]:
            textpat=f'*{pat}*.npy'
            #print(textpat)
            count=count+sum(1 for item in paths if fnmatch.fnmatch(item, textpat))
            #print(count)
        class_counts_split[i,j]=count

class_counts_split = class_counts_split.astype(int)

np.sum(class_counts_split)-np.sum(class_counts)

# Augmentation params
omit_index = [0,3]  # do not include broadband tremor and non-tremor signal in count determination
noise_index = 6  # use noise samples to augment
testval_ratio = 0.2  # use 20% of sparse-est class count to pull test and validation sets
noise_ratio = 0.35  # weight of noise sample added for augmentation

# Configure train, validation and test paths and determine unique classes
train_paths, valid_paths, test_paths = augment_labeled_dataset(npy_dir=SPEC_DIR, omit_index=omit_index,
                                                             noise_index=noise_index,testval_ratio=testval_ratio,
                                                             noise_ratio=noise_ratio)
train_classes = [int(i.split("_")[-1][0]) for i in train_paths]
unique_classes = np.unique(train_classes)

# Count the number of samples of each augmented class
train_counts = np.unique(train_classes, return_counts=True)[1]

#%% now plot

x_labels = []
for item in label_dict.keys():
    if ' ' in item:
        split_items = item.split(' ')
        x_labels.append(f'{split_items[0]}\n{split_items[1]}')
    else:
        x_labels.append(item)

# plot stacked bar chart of class counts
#class_counts_all = {
#    "Pavlof": class_counts_pavlof,
#    "Semi": class_counts_semi,
#}
width = 0.75
bottom = np.zeros(nclasses)
co = ['darkgray', 'darkred']


fig, ax = plt.subplots()
fig.set_size_inches(7, 5.25)
#plt.clf()
i=0
#for boolean, class_count_tmp in VOLC_STA_WC.items():
for i,key in enumerate(VOLC_STA_WC):
    p = ax.bar(label_dict.keys(), class_counts_split[i,:], width, label=key,
               bottom=bottom, color=co[i])
    bottom += class_counts_split[i,:]
    #i=i+1
ax.set_ylabel('Number of Samples')
ax.set_xticks(np.array(list(label_dict.values())))
ax.set_xticklabels(x_labels)
ax.legend(VOLC_STA_WC.keys())


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

