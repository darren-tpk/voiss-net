# Import dependencies
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import colorcet as cc

# Point to directory
metrics_dir = '/Users/darrentpk/Desktop/GitHub/tremor_ml/metrics/PS4A_BDF_metrics/'

# Read
tmpl = np.load(metrics_dir + 'tmpl_all_20210101_20230101.npy')
rmsp = np.load(metrics_dir + 'rmsp_all_20210101_20230101.npy')
pe = np.load(metrics_dir + 'pe_all_20210101_20230101.npy')
fc = np.load(metrics_dir + 'fc_all_20210101_20230101.npy')
fd = np.load(metrics_dir + 'fd_all_20210101_20230101.npy')
fsd = np.load(metrics_dir + 'fsd_all_20210101_20230101.npy')
spec_db = np.load(metrics_dir + 'spec_db_all_20210101_20230101.npy')

# Define spectrogram params
starttime = UTCDateTime(2021,1,1)
endtime = UTCDateTime(2023,1,1)
freq_lims = (0.5,10)
cmap = cc.cm.rainbow
time_tick_list = [UTCDateTime(2021,i,1) for i in range(1,13)] + \
                 [UTCDateTime(2022,i,1) for i in range(1,13)]
time_tick_list_mpl = [t.matplotlib_date for t in time_tick_list]
time_tick_labels = [time.strftime('%m/%y') for time in time_tick_list]

# Compute network median
tmpl = tmpl[0,:]  # only need 1 row for time
rmsp = rmsp[2,:]  # only use PS4A RMS Pressure
option = 'PS4A'  # or PS4A
if option == 'network':
    pe_nm = np.nanmedian(pe, axis=0)
    fc_nm = np.nanmedian(fc, axis=0)
    fd_nm = np.nanmedian(fd, axis=0)
    fsd_nm = np.nanmedian(fsd, axis=0)
elif option == 'PS4A':
    pe_nm = pe[2,:]
    fc_nm = fc[2,:]
    fd_nm = fd[2,:]
    fsd_nm = fsd[2,:]

# Execute median filter on network median
KERNEL_SIZE = 7
rmsp_mf = medfilt(rmsp, kernel_size=KERNEL_SIZE)
pe_mf = medfilt(pe_nm, kernel_size=KERNEL_SIZE)
fc_mf = medfilt(fc_nm, kernel_size=KERNEL_SIZE)
fd_mf = medfilt(fd_nm, kernel_size=KERNEL_SIZE)
fsd_mf = medfilt(fsd_nm, kernel_size=KERNEL_SIZE)

# Compute median of median-filtered network medians
rmsp_mm = np.nanmedian(rmsp_mf)
pe_mm = np.nanmedian(pe_mf)
fc_mm = np.nanmedian(fc_mf)
fd_mm = np.nanmedian(fd_mf)
fsd_mm = np.nanmedian(fsd_mf)

# Plot spectrogram and time series of metrics
figsize = (32, 11)
fig = plt.figure(figsize=figsize)
gs_top = plt.GridSpec(6, 1, top=0.89)
gs_base = plt.GridSpec(6, 1, hspace=0)
ax1 = fig.add_subplot(gs_top[0, :])
ax2 = fig.add_subplot(gs_base[1, :])
other_axes = [fig.add_subplot(gs_base[i, :], sharex=ax2) for i in range(2, 6)]
axs = [ax2] + other_axes
for ax in axs[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
c = ax1.imshow(spec_db, extent=[starttime.matplotlib_date, endtime.matplotlib_date, freq_lims[0], freq_lims[-1]],
               vmin=np.nanpercentile(spec_db, 20), vmax=np.nanpercentile(spec_db, 97.5), origin='lower', aspect='auto',
               interpolation=None, cmap=cmap)
ax1.set_ylim([freq_lims[0], freq_lims[1]])
ax1.tick_params(axis='y', labelsize=18)
ax1.set_xlim([starttime.matplotlib_date, endtime.matplotlib_date])
ax1.set_xticks(time_tick_list_mpl)
ax1.set_xticklabels([], fontsize=22)
ax1.set_yticks([0.5,5,10])
ax1.set_yticklabels(['0.5','5','10'],fontsize=18)
ax1.set_ylabel('PS4A.BDF\nFrequency',fontsize=22)
axs[0].plot(tmpl, rmsp, color='k', alpha=0.8)
axs[0].axhline(y=rmsp_mm, color='r', linewidth=2.5, linestyle='--')
axs[0].set_ylabel('$Pa_{RMS}$', fontsize=22)
axs[0].set_ylim([0,50])
axs[1].plot(tmpl, pe_mf, color='k', alpha=0.85)
axs[1].axhline(y=pe_mm, color='r',linewidth=2.5, linestyle='--')
axs[1].set_ylabel('$p_e$', fontsize=24)
axs[2].plot(tmpl, fc_mf, color='k', alpha=0.8)
axs[2].axhline(y=fc_mm,color='r',linewidth=2.5, linestyle='--')
axs[2].set_ylabel('$f_c$ (Hz)', fontsize=22)
axs[2].set_ylim([3,7])
axs[3].plot(tmpl, fd_mf, color='k', alpha=0.8)
axs[3].axhline(y=fd_mm,color='red',linewidth=2.5, linestyle='--')
axs[3].set_ylabel('$f_d$ (Hz)', fontsize=22)
axs[3].set_ylim([0.9,4])
axs[4].plot(tmpl, fsd_mf, color='k', alpha=0.8)
axs[4].axhline(y=fsd_mm, color='r', linewidth=2.5, linestyle='--')
axs[4].set_ylabel('$\sigma_f$ (Hz)', fontsize=22)
axs[4].set_ylim([0,10])
time_tick_labels2 = [time.strftime('%b \'%y') for time in time_tick_list]
for j in range(5):
    axs[j].set_xlim([starttime.matplotlib_date, endtime.matplotlib_date])
    axs[j].tick_params(axis='y', labelsize=18)
    axs[j].set_xticks(time_tick_list_mpl)
    axs[j].set_xticklabels([])
axs[-1].set_xticklabels(time_tick_labels2, fontsize=20, rotation=30)
axs[-1].set_xlabel('Time', fontsize=28)
# axs[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,7))
# axs[0].get_yaxis().get_offset_text().set_visible(False)
# axs[0].annotate(r'$\times$10$^{7}}$', xy=(.001, .75), xycoords='axes fraction', fontsize=18)
# plt.savefig('/Users/darrentpk/Desktop/GitHub/tremor_ml/infrasound_metrics.png',bbox_inches='tight',transparent=False)
fig.show()