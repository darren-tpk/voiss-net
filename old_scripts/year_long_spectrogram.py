# Import packages
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from obspy import UTCDateTime

# Load data
spec_db_all = np.load('/Users/darrentpk/Desktop/IAVCEI 2023/poster_media/spec_db_all.npy')
trace_time_matplotlib = np.load('/Users/darrentpk/Desktop/IAVCEI 2023/poster_media/tt_mpl_all.npy')

# Define variables
freq_lims = (0.5,10)
T1 = UTCDateTime(2021,1,1)
T2 = UTCDateTime(2023,1,1)
sample_frequencies = np.arange(0,25.1,0.1)
cmap = cc.cm.rainbow

# plot it
# Prepare time ticks for spectrogram and waveform figure (Note that we divide the duration into 10 equal increments and 11 uniform ticks)
time_tick_list = []
for i in range(1,13):
    month_utcdatetime = UTCDateTime(2021,i,1)
    time_tick_list.append(month_utcdatetime)
for i in range(1,13):
    month_utcdatetime = UTCDateTime(2022,i,1)
    time_tick_list.append(month_utcdatetime)
time_tick_list_mpl = [t.matplotlib_date for t in time_tick_list]
time_tick_labels = [time.strftime('%m/%y') for time in time_tick_list]

fig, ax1 = plt.subplots(figsize=(32,2))
# Determine frequency limits and trim spec_db
freq_min = freq_lims[0]
freq_max = freq_lims[1]
spec_db_plot = spec_db_all[np.flatnonzero((sample_frequencies > freq_min) & (sample_frequencies < freq_max)), :]
c = ax1.imshow(spec_db_all, extent=[trace_time_matplotlib[0], trace_time_matplotlib[-1], sample_frequencies[0],
                                sample_frequencies[-1]],
               vmin=-170.59161262634825,
               vmax=-128.7013728179577,
               origin='lower', aspect='auto', interpolation=None, cmap=cmap)
ax1.set_ylim([freq_min, freq_max])
ax1.tick_params(axis='y', labelsize=18)
ax1.set_xlim([T1.matplotlib_date, T2.matplotlib_date])
ax1.set_xticks(time_tick_list_mpl)
ax1.set_xticklabels([])
ax1.set_yticks([0.5,5,10])
ax1.set_yticklabels(['0.5','5','10'],fontsize=22)
ax1.set_ylabel('PS1A.BHZ\nFrequency',fontsize=24)
# ax1.set_title('AV.PS1A..BHZ', fontsize=24, fontweight='bold')
# fig.savefig('/Users/darrentpk/Desktop/year_spec.png', bbox_inches='tight', transparent=True)
fig.show()

