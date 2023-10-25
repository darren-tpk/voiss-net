import matplotlib.pyplot as plt

# Plot seismic
initial_cc = [4106, 674, 3828, 9279, 808, 15849]
balance_cc = [0, 3838+134*2, 3838+134*2, 0, 3838+134*2, 0]
val_cc = [134*2, 134*2, 134*2, 134*2, 134*2, 134*2]
test_cc = [134, 134, 134, 134, 134, 134]

# Plot bar chart with data points
fig, ax = plt.subplots(figsize=(5,5))
ax.bar(range(6), balance_cc, color='silver', label='Augmented Dataset')
ax.bar(range(6), initial_cc, color='dimgrey', label='Labeled Dataset')
ax.bar(range(6), val_cc, color='salmon', label='Validation Set')
ax.bar(range(6), test_cc, color='red', label='Test Set')
ax.axhline(y=3838+134*2, linestyle='--', color='black')
ax.set_ylabel('Class Counts')
ax.set_xlabel('Class')
ax.set_xticks(range(6))
ax.set_xticklabels(['Broadband\nTremor','Harmonic\nTremor','Mono\nTremor','Non-Tremor\nSignal',
                    'Explosion','Noise'],fontsize=8)
ax.set_title('Seismic Class Balancing')
ax.legend()
# fig.savefig('/Users/darrentpk/Desktop/seis_bal.png', bbox_inches='tight')
fig.show()


import matplotlib.pyplot as plt

# Plot infrasound
initial_cc = [1666, 441, 14088, 2612]
balance_cc = [2436+88*2, 2436+88*2, 0, 0]
val_cc = [88*2, 88*2, 88*2, 88*2]
test_cc = [88, 88, 88, 88]

# Plot bar chart with data points
fig, ax = plt.subplots(figsize=(4,5))
ax.bar(range(4), balance_cc, color='silver', label='Augmented Dataset')
ax.bar(range(4), initial_cc, color='dimgrey', label='Labeled Dataset')
ax.bar(range(4), val_cc, color='salmon', label='Validation Set')
ax.bar(range(4), test_cc, color='red', label='Test Set')
ax.axhline(y=2436+88*2, linestyle='--', color='black')
ax.set_ylabel('Class Counts')
ax.set_xlabel('Class')
ax.set_xticks(range(4))
ax.set_xticklabels(['Infrasound\nTremor','Explosion','Wind Noise','Electronic\nNoise'],fontsize=9)
ax.set_title('Infrasound Class Balancing')
# ax.legend()
# fig.savefig('/Users/darrentpk/Desktop/infra_bal.png', bbox_inches='tight')
fig.show()
