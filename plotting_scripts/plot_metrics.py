# Make and save big time series
from waveform_collection import gather_waveforms
from toolbox import compute_metrics
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

###### load and plot stuff
output_dir = '/Users/darrentpk/Desktop/GitHub/tremor_ml/metrics/'

for kernel_size in [7]:
    tmpl = np.load(output_dir + 'tmpl_20210101_20230101.npy')
    tmpl2 = np.load(output_dir + 'tmpl_all.npy')
    dr = np.load(output_dir + 'dr_all.npy')
    pe = np.load(output_dir + 'pe_20210101_20230101.npy')
    fc = np.load(output_dir + 'fc_20210101_20230101.npy')
    fd = np.load(output_dir + 'fd_20210101_20230101.npy')
    fsd = np.load(output_dir + 'fsd_20210101_20230101.npy')
    dr = medfilt(dr, kernel_size=kernel_size)
    pe = medfilt(pe, kernel_size=kernel_size)
    fc = medfilt(fc, kernel_size=kernel_size)
    fd = medfilt(fd, kernel_size=kernel_size)
    fsd = medfilt(fsd, kernel_size=kernel_size)

    fig, axs = plt.subplots(5, 1, figsize=(32, 8))
    fig.subplots_adjust(hspace=0)
    axs[0].plot(tmpl2, dr, color='k', alpha=0.8)
    axs[0].set_ylabel('$D_R$ (cm$^2$)', fontsize=22)
    axs[0].set_ylim([0,5])
    axs[0].axhline(y=np.nanmedian(dr), color='red', linewidth=2.5, linestyle='--')
    axs[1].plot(tmpl, pe, color='k', alpha=0.85)
    axs[1].set_ylabel('$p_e$', fontsize=24)
    axs[1].axhline(y=np.nanmedian(pe),color='red',linewidth=2.5, linestyle='--')
    axs[2].plot(tmpl, fc, color='k', alpha=0.8)
    axs[2].set_ylabel('$f_c$ (Hz)', fontsize=22)
    axs[2].set_ylim([2.25,6])
    axs[2].axhline(y=np.nanmedian(fc),color='red',linewidth=2.5, linestyle='--')
    axs[3].plot(tmpl, fd, color='k', alpha=0.8)
    axs[3].set_ylabel('$f_d$ (Hz)', fontsize=22)
    axs[3].set_ylim([0.75,4])
    axs[3].axhline(y=np.nanmedian(fd),color='red',linewidth=2.5, linestyle='--')
    axs[4].plot(tmpl, fsd, color='k', alpha=0.8)
    axs[4].set_ylabel('$\sigma_f$ (Hz)', fontsize=22)
    axs[4].set_ylim([0,5e-4])
    axs[4].axhline(y=np.nanmedian(fsd),color='red',linewidth=2.5, linestyle='--')
    starttime = UTCDateTime(2021,1,1)
    endtime = UTCDateTime(2023,1,1)
    denominator = 24
    time_tick_list = np.arange(starttime, endtime, (endtime - starttime) / denominator)
    time_tick_list_mpl = [t.matplotlib_date for t in time_tick_list]
    time_tick_labels = [time.strftime('%b \'%y') for time in time_tick_list]
    for j in range(5):
        axs[j].set_xlim([starttime.matplotlib_date, endtime.matplotlib_date])
        axs[j].tick_params(axis='y', labelsize=18)
        axs[j].set_xticks(time_tick_list_mpl)
        axs[j].set_xticklabels([])
    axs[-1].set_xticklabels(time_tick_labels, fontsize=20, rotation=30)
    axs[-1].set_xlabel('Time', fontsize=28)
    axs[-1].ticklabel_format(axis='y', style='scientific', scilimits=(-5,-5))
    axs[-1].get_yaxis().get_offset_text().set_visible(False)
    axs[-1].annotate(r'$\times$10$^{-5}}$', xy=(.001, .75), xycoords='axes fraction', fontsize=18)
    # plt.savefig('/Users/darrentpk/Desktop/timeline9.png',bbox_inches='tight',transparent=True)
    fig.show()

