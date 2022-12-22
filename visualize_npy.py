
# Import dependencies
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

# Define numpy directory
npy_dir = '/Users/darrentpk/Desktop/PVV_npy/'

# Prepare label dictionary
label_dict = {'Broadband Tremor': 1,
              'Harmonic Tremor': 2,
              'Monochromatic Tremor': 3,
              'Non-tremor Signal': 4,
              'Explosion': 5,
              'Noise': 6}

labels = ['Broadband Tremor', 'Harmonic Tremor', 'Monochromatic Tremor', 'Non-tremor Signal',
          'Explosion', 'Noise']

# Define conditions
station_check = 'PVV'
# PN7A, PS1A, PS4A, PV6A, PVV

for label_check in labels:

    # Filter filenames
    all_filenames = glob.glob(npy_dir + station_check + '*' + str(label_dict[label_check]) + '.npy')

    # Plot 25
    N = 16

    # Randomly sample N files from list
    chosen_filenames = random.sample(all_filenames, N)

    # Craft title
    fig, axs = plt.subplots(nrows=int(np.sqrt(N)), ncols=int(np.sqrt(N)), figsize=(6,8))
    fig.suptitle('%d samples of %s slices on %s' % (N, label_check, station_check), fontweight='bold')
    for i in range(int(np.sqrt(N))):
        for j in range(int(np.sqrt(N))):
            filename_index = i * int(np.sqrt(N)) + (j + 1) - 1
            if filename_index > (len(chosen_filenames)-1):
                continue
            spec_db = np.load(chosen_filenames[filename_index])
            if np.sum(spec_db < -250) > 0:
                print(i, j)
            axs[i, j].imshow(spec_db, vmin=np.percentile(spec_db, 20), vmax=np.percentile(spec_db, 97.5),
                           origin='lower', aspect='auto', interpolation=None, cmap=cc.cm.rainbow)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    fig.show()