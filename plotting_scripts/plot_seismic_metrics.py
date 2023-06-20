# Import dependencies
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import colorcet as cc

# Point to directory
metrics_dir = '/Users/darrentpk/Desktop/GitHub/tremor_ml/metrics/PS1A_BHZ_metrics/'

# Read
tag = '_20210101_20230101'
tmpl = np.load(metrics_dir + 'tmpl_all' + tag + '.npy')
dr = np.load(metrics_dir + 'dr_all' + tag + '.npy')
# dr = dr * (EXPONENTIAL Q FACTOR -- see Haney et al. on Bogoslof exp(πf(D−10)/cQ)),
pe = np.load(metrics_dir + 'pe_all' + tag + '.npy')
fc = np.load(metrics_dir + 'fc_all' + tag + '.npy')
fd = np.load(metrics_dir + 'fd_all' + tag + '.npy')
fsd = np.load(metrics_dir + 'fsd_all' + tag + '.npy')
spec_db = np.load(metrics_dir + 'spec_db_all' + tag + '.npy')

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
option = 'network'  # or PS4A
if option == 'network':
    dr_nm = np.nanmedian(dr, axis=0)
    pe_nm = np.nanmedian(pe, axis=0)
    fc_nm = np.nanmedian(fc, axis=0)
    fd_nm = np.nanmedian(fd, axis=0)
    fsd_nm = np.nanmedian(fsd, axis=0)
elif option == 'PS1A':
    dr_nm = dr[1,:]
    pe_nm = pe[1,:]
    fc_nm = fc[1,:]
    fd_nm = fd[1,:]
    fsd_nm = fsd[1,:]

# Execute median filter on network median
KERNEL_SIZE = 7
dr_mf = medfilt(dr_nm, kernel_size=KERNEL_SIZE)
pe_mf = medfilt(pe_nm, kernel_size=KERNEL_SIZE)
fc_mf = medfilt(fc_nm, kernel_size=KERNEL_SIZE)
fd_mf = medfilt(fd_nm, kernel_size=KERNEL_SIZE)
fsd_mf = medfilt(fsd_nm, kernel_size=KERNEL_SIZE)

# Compute median of median-filtered network medians
dr_mm = np.nanmedian(dr_mf)
pe_mm = np.nanmedian(pe_mf)
fc_mm = np.nanmedian(fc_mf)
fd_mm = np.nanmedian(fd_mf)
fsd_mm = np.nanmedian(fsd_mf)

# Prepare avo color code markers
tick_numbers = [(t-starttime)/86400 for t in time_tick_list+[endtime]]
g2y_number = (UTCDateTime(2021,7,9)-starttime)/86400
y2o_number = (UTCDateTime(2021,8,5)-starttime)/86400
o2y_number = (UTCDateTime(2022,12,17)-starttime)/86400
y2e_number = (endtime-starttime)/86400

# Plot spectrogram and time series of metrics
figsize = (32, 17)
fig = plt.figure(figsize=figsize)
gs_top = plt.GridSpec(7, 1, height_ratios=[2,7,7,7,7,7,7], top=0.82)
gs_middle = plt.GridSpec(7, 1, top=0.82)
gs_base = plt.GridSpec(7, 1, hspace=0)
ax0 = fig.add_subplot(gs_top[0, :])
ax1 = fig.add_subplot(gs_top[1, :])
ax2 = fig.add_subplot(gs_base[2, :])
axs2 = [ax2] + [fig.add_subplot(gs_base[i, :], sharex=ax2) for i in range(3, 7)]
for ax in axs2[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax0.axvspan(0, g2y_number, color='green')
ax0.axvspan(g2y_number, y2o_number, color='yellow')
ax0.axvspan(y2o_number, o2y_number, color='orange')
ax0.axvspan(o2y_number, y2e_number, color='yellow')
ax0.set_xlim([0,y2e_number])
ax0.set_xticks(tick_numbers)
ax0.set_xticklabels([])
ax0.set_yticks([])
ax0.set_ylabel('AVO', fontsize=27, rotation=0, labelpad=45, verticalalignment='center')
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
ax1.set_ylabel('PS1A.BHZ\nFrequency',fontsize=22)
axs2[0].plot(tmpl, dr_mf, color='k', alpha=0.8)
axs2[0].axhline(y=dr_mm, color='r', linewidth=2.5, linestyle='--')
axs2[0].set_ylabel('$D_R$', fontsize=22)
axs2[0].set_ylim([0,5])
axs2[1].plot(tmpl, pe_mf, color='k', alpha=0.85)
axs2[1].axhline(y=pe_mm, color='r',linewidth=2.5, linestyle='--')
axs2[1].set_ylabel('$p_e$', fontsize=24)
axs2[2].plot(tmpl, fc_mf, color='k', alpha=0.8)
axs2[2].axhline(y=fc_mm,color='r',linewidth=2.5, linestyle='--')
axs2[2].set_ylabel('$f_c$ (Hz)', fontsize=22)
axs2[2].set_ylim([2,6])
axs2[3].plot(tmpl, fd_mf, color='k', alpha=0.8)
axs2[3].axhline(y=fd_mm,color='red',linewidth=2.5, linestyle='--')
axs2[3].set_ylabel('$f_d$ (Hz)', fontsize=22)
axs2[3].set_ylim([0.5,4])
axs2[4].plot(tmpl, fsd_mf, color='k', alpha=0.8)
axs2[4].axhline(y=fsd_mm, color='r', linewidth=2.5, linestyle='--')
axs2[4].set_ylabel('$\sigma_f$ (Hz)', fontsize=22)
axs2[4].set_ylim([0,12])
time_tick_labels2 = [time.strftime('%b \'%y') for time in time_tick_list]
for j in range(5):
    axs2[j].set_xlim([starttime.matplotlib_date, endtime.matplotlib_date])
    axs2[j].tick_params(axis='y', labelsize=18)
    axs2[j].set_xticks(time_tick_list_mpl)
    axs2[j].set_xticklabels([])
axs2[-1].set_xticklabels(time_tick_labels2, fontsize=20, rotation=30)
axs2[-1].set_xlabel('Time', fontsize=28)
# axs2[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,7))
# axs2[0].get_yaxis().get_offset_text().set_visible(False)
# axs2[0].annotate(r'$\times$10$^{7}}$', xy=(.001, .75), xycoords='axes fraction', fontsize=18)
plt.savefig('./figures/seismic_metrics.png',bbox_inches='tight',transparent=False)
fig.show()