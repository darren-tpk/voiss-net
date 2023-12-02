# Plot multidisciplinary plot, top to bottom order
# 1. AVO color code
# 2. PS1A Spectrogram (Seismic)
# 3. Barcode Plot (Seismic)
# 4. Reduced displacement (Seismic)
# 5. Barcode plot (Infrasound)
# 6. Spaced-based observations (Hotspot & SO2)
# 7. Explosion time series

# Import all dependencies
import pickle
import pandas as pd
import numpy as np
import colorcet as cc
from obspy import UTCDateTime
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import medfilt

# Plot settings
starttime = UTCDateTime(2021,1,1)
endtime = UTCDateTime(2023,3,1)
common_norm = True
multiple = 'fill'
bw_method = 1/10000
outfile = '/Users/darrentpk/Desktop/multidisciplinary_fig_v6.png'

# Define time ticks on x-axis
month_utcdatetimes = []
month_utcdate = starttime
while month_utcdate <= endtime:
    month_utcdatetimes.append(month_utcdate)
    if month_utcdate.month + 1 <= 12:
        month_utcdate += UTCDateTime(month_utcdate.year, month_utcdate.month + 1, 1) - month_utcdate
    else:
        month_utcdate += UTCDateTime(month_utcdate.year + 1, 1, 1) - month_utcdate
xticklabels_horiz = [t.strftime('%b \'%y') for t in month_utcdatetimes]
total_days = (endtime-starttime)/86400
xticks_horiz = [(t - starttime)/86400/total_days for t in month_utcdatetimes]

# Define labeled start and end
labeled_start = UTCDateTime(2021, 7, 22)
labeled_end = UTCDateTime(2021, 9, 22)
labeled_start_pct_since = (UTCDateTime(2021, 7, 22) - starttime)/86400/total_days
labeled_end_pct_since = (UTCDateTime(2021, 9, 22) - starttime)/86400/total_days

# Define AVO color code axvspan limits
g2y_pct_since = (UTCDateTime(2021,7,9)-starttime)/86400/total_days
y2o_pct_since = (UTCDateTime(2021,8,5)-starttime)/86400/total_days
o2y_pct_since = (UTCDateTime(2022,12,17)-starttime)/86400/total_days
y2g_pct_since = (UTCDateTime(2023,1,19)-starttime)/86400/total_days
g2e_pct_since = (month_utcdatetimes[-1]-starttime)/86400/total_days

# Read spectrogram and reduced displacement data
metrics_dir = './metrics/PS1A_BHZ_metrics/'
tag = '_20210101_20230301'
tmpl = np.load(metrics_dir + 'tmpl_all' + tag + '.npy')
tmpl = tmpl[0,:]  # only need 1 row for time
dr = np.load(metrics_dir + 'dr_all' + tag + '.npy')
dr_nm = np.nanmedian(dr, axis=0)  # network median
dr_mf = medfilt(dr_nm, kernel_size=7)  # median filter
spec_db = np.load(metrics_dir + 'spec_db_all' + tag + '.npy')

# Parse seismic dataframe
daily_bin_df = pd.read_csv('./data_frames/daily_bin_df.csv', index_col=0)
daily_bin_df['bb_alpha'] = np.sqrt(daily_bin_df['bb_count'] / 1440)
daily_bin_df['ht_alpha'] = np.sqrt(daily_bin_df['ht_count'] / 1440)
daily_bin_df['mt_alpha'] = np.sqrt(daily_bin_df['mt_count'] / 1440)
daily_bin_df['nt_alpha'] = np.sqrt(daily_bin_df['nt_count'] / 1440)
daily_bin_df['exp_alpha'] = np.sqrt(daily_bin_df['exp_count'] / 1440)
daily_bin_df['noise_alpha'] = np.sqrt(daily_bin_df['noise_count'] / 1440)
matrix_plot = np.ones((6, len(daily_bin_df)))
for i in range(6):
    matrix_plot[i, :] = matrix_plot[i, :] * i
matrix_plot = np.flip(matrix_plot, axis=0)
matrix_alpha = np.vstack((daily_bin_df['bb_alpha'], daily_bin_df['ht_alpha'], daily_bin_df['mt_alpha'], daily_bin_df['nt_alpha'], daily_bin_df['exp_alpha'], daily_bin_df['noise_alpha']))
matrix_alpha = np.flip(matrix_alpha, axis=0)
rgb_values = np.array([
    [193, 39, 45],
    [0, 129, 118],
    [0, 0, 167],
    [238, 204, 22],
    [164, 98, 0],
    [40, 40, 40]])
rgb_keys = ['BT',
            'HT',
            'MT',
            'EQ',
            'EX',
            'NO']
rgb_ratios = rgb_values / 255
cmap = ListedColormap(rgb_ratios)
amap = np.ones([256, 4])
amap[:, 3] = np.linspace(0, 1, 256)
amap = ListedColormap(amap)

# Load thermal data
thermal_filepath = './data_frames/thermal_anomalies.csv'
thermal = pd.read_csv(thermal_filepath)
rp_time = list(thermal['image_time'])
rp_pct_since = np.array([(UTCDateTime(t)-starttime)/86400/total_days for t in rp_time])
nodata_pct_since = (UTCDateTime(2023,1,1)-starttime)/86400/total_days
text_pct_since = (UTCDateTime(2023,2,1)-starttime)/86400/total_days
rp_values = np.array(list(thermal['Radiative Power (MW)']))

# Load SO2 data
so2_filepath = './data_frames/so2_em_rate.csv'
so2 = pd.read_csv(so2_filepath)
so2_date = list(so2['date'])
so2_pct_since = np.array([((UTCDateTime(t)-starttime)/86400 + 0.5)/total_days  for t in so2_date])
so2_rates = np.array(list(so2['em_rate']))

# Read infrasound spectrogram
metrics_dir_inf = './metrics/PS4A_BDF_metrics/'
tmpl_inf = np.load(metrics_dir_inf + 'tmpl_all' + tag + '.npy')
tmpl_inf = tmpl_inf[0,:]  # only need 1 row for time
spec_db_inf = np.load(metrics_dir_inf + 'spec_db_all' + tag + '.npy')

# Parse infra dataframe
daily_bin_df_inf = pd.read_csv('./data_frames/daily_bin_df_infra.csv', index_col=0)
daily_bin_df_inf['it_alpha'] = np.sqrt(daily_bin_df_inf['it_count'] / 1440)
daily_bin_df_inf['exp_alpha'] = np.sqrt(daily_bin_df_inf['exp_count'] / 1440)
daily_bin_df_inf['wn_alpha'] = np.sqrt(daily_bin_df_inf['wn_count'] / 1440)
daily_bin_df_inf['en_alpha'] = np.sqrt(daily_bin_df_inf['en_count'] / 1440)
matrix_plot_inf = np.ones((4, len(daily_bin_df_inf)))
for i in range(4):
    matrix_plot_inf[i, :] = matrix_plot_inf[i, :] * i
matrix_plot_inf = np.flip(matrix_plot_inf, axis=0)
matrix_alpha_inf = np.vstack((daily_bin_df_inf['it_alpha'], daily_bin_df_inf['exp_alpha'],
                              daily_bin_df_inf['wn_alpha'], daily_bin_df_inf['en_alpha']))
matrix_alpha_inf = np.flip(matrix_alpha_inf, axis=0)
rgb_values_inf = np.array([
    [103,  52, 235],
    [235, 152,  52],
    [ 40,  40,  40],
    [ 15,  37,  60]])
rgb_keys_inf = ['IT',
                'EX',
                'WN',
                'EN']
rgb_ratios_inf = rgb_values_inf / 255
cmap_inf = ListedColormap(rgb_ratios_inf)

# # Craft explosion dataframe
# exp_bool = np.logical_and(matrix_condensed==4, matrix_condensed_inf==1)
# exp_times = [starttime + (i*60) for i in range(len(exp_bool)) if exp_bool[i]==True]
# seis_exp_pnorm = matrix_pnorm[exp_bool==True]
# infra_exp_pnorm = matrix_pnorm_inf[exp_bool==True]
# df_exp = pd.DataFrame(list(zip(exp_times, seis_exp_pnorm, infra_exp_pnorm)), columns =['time', 'pnorm_seis','pnorm_infra'])
# df_exp.to_csv('./data_frames/exp_df_pnorm.csv')

# Parse explosion dataframes
df_exp = pd.read_csv('./data_frames/exp_df_pnorm.csv', usecols=range(1,4))
best_df_exp = df_exp[(df_exp['pnorm_seis']>=.6) & (df_exp['pnorm_infra']>=.6)]
best_exp_times = [UTCDateTime(t) for t in best_df_exp['time'].tolist()]
best_exp_pct_since = [(t-starttime)/86400/total_days for t in best_exp_times]
rtm_df = pd.read_csv('./data_frames/pavlof_rtm.csv')
rtm_dates = rtm_df['Date'].tolist()
rtm_dists = rtm_df['Distance (M)'].tolist()
rtm_amps = rtm_df['Stack Amplitude'].tolist()
rtm_times = [UTCDateTime(rtm_dates[j]) for j in range(len(rtm_dates)) if ((rtm_dists[j] < 600) and rtm_amps[j]>0.9)]
rtm_pct_since = [(t-starttime)/86400/total_days for t in rtm_times]

# Commence multidisciplinary plotting
figsize = (37, 19)
height_ratios=[1.5,7,9,7,7,7,6,2]
fig, axs = plt.subplots(8, 1, figsize=figsize, gridspec_kw={'height_ratios': height_ratios})

# Plot AVO color code
axs[0].axvspan(0, g2y_pct_since, color='green')
axs[0].axvspan(g2y_pct_since, y2o_pct_since, color='yellow')
axs[0].axvspan(y2o_pct_since, o2y_pct_since, color='orange')
axs[0].axvspan(o2y_pct_since, y2g_pct_since, color='yellow')
axs[0].axvspan(y2g_pct_since, g2e_pct_since, color='green')
axs[0].set_xlim([0,g2e_pct_since])
axs[0].set_xticks(xticks_horiz)
axs[0].set_xticklabels([])
axs[0].set_yticks([])
axs[0].set_ylabel('AVO', rotation=0, labelpad=40, fontsize=27, verticalalignment='center')

# Plot spectrogram
freq_lims = (0.5,10)
axs[1].imshow(spec_db, extent=[xticks_horiz[0], xticks_horiz[-1], freq_lims[0], freq_lims[-1]],
              vmin=np.nanpercentile(spec_db, 20), vmax=np.nanpercentile(spec_db, 97.5), origin='lower', aspect='auto',
              interpolation='None', cmap=cc.cm.rainbow)
axs[1].set_ylim([freq_lims[0], freq_lims[1]])
axs[1].tick_params(axis='y', labelsize=18)
axs[1].set_xlim([xticks_horiz[0], xticks_horiz[-1]])
axs[1].set_xticks(xticks_horiz)
axs[1].set_xticklabels([], fontsize=22)
axs[1].set_yticks([0.5,5,10])
axs[1].set_yticklabels(['0.5','5','10'],fontsize=18)
axs[1].set_ylabel('PS1A.BHZ\nFrequency',fontsize=22)

# Plot kernel plot
axs[2].imshow(matrix_plot, cmap=cmap, interpolation='None', extent=[xticks_horiz[0], xticks_horiz[-1], 0, 6], aspect='auto', origin='lower')
axs[2].imshow(1-matrix_alpha, cmap=amap, interpolation='None', extent=[xticks_horiz[0], xticks_horiz[-1], 0, 6], aspect='auto', origin='lower')
axs[2].set_yticks(np.arange(0.5, 6.5, 1))
axs[2].set_yticklabels(np.flip(rgb_keys))
axs[2].set_xticks(xticks_horiz)
axs[2].set_xticklabels([])
axs[2].set(xlabel=None)
axs[2].set_xlim([xticks_horiz[0], xticks_horiz[-1]])
axs[2].set_ylabel('Seismic\nClass',fontsize=22)
axs[2].tick_params(axis='y', labelsize=18)

# Plot reduced displacement
axs[3].plot(tmpl, dr_mf, color='k', alpha=0.8)
axs[3].set_ylabel('$D_R$ (cm$^2$)', fontsize=22)
axs[3].set_ylim([0,5])
axs[3].set_xlim([month_utcdatetimes[0].matplotlib_date, month_utcdatetimes[-1].matplotlib_date])
axs[3].tick_params(axis='y', labelsize=18)
axs[3].set_xticks([mutc.matplotlib_date for mutc in month_utcdatetimes])
axs[3].set_xticklabels([])

# Plot Thermal data and SO2 detections
axs[4].plot(rp_pct_since, rp_values, color='k', alpha=0.8)
axs[4].axvspan(xmin=nodata_pct_since,xmax=g2e_pct_since, color='lightgrey',alpha=0.5)
axs[4].text(x=text_pct_since, y=110, s='Not\nComputed', fontsize=20, horizontalalignment='center', verticalalignment='center')
axs[4].set_ylabel('Radiative\nPower (MW)', fontsize=22)
axs[4].set_xlim([xticks_horiz[0], xticks_horiz[-1]])
axs[4].set_xticks(xticks_horiz)
axs[4].set_xticklabels([])
axs[4].set_ylim([0,220])
axs[4].set_yticks(np.arange(0,220,100))
axs[4].tick_params(axis='y', labelsize=18)
axs4b = axs[4].twinx()
axs4b.set_ylabel('SO2 Rate (t/day)', fontsize=22, color='red')
for i in range(len(so2_pct_since)):
    axs4b.axvline(x=so2_pct_since[i], ymax=so2_rates[i]/330, linewidth=2.5, color='r', alpha=0.8)
axs4b.plot(so2_pct_since, so2_rates, 'r^')
axs4b.set_yticks(np.arange(100,330,100))
axs4b.tick_params(axis='y', labelsize=18, direction='in', pad=-40, labelcolor='red')
axs4b.spines['right'].set_color('darkred')

# Plot infrasound spectrogram
freq_lims = (0.5,10)
axs[5].imshow(spec_db_inf, extent=[xticks_horiz[0], xticks_horiz[-1], freq_lims[0], freq_lims[-1]],
              vmin=np.nanpercentile(spec_db_inf, 20), vmax=np.nanpercentile(spec_db_inf, 97.5), origin='lower', aspect='auto',
              interpolation='None', cmap=cc.cm.rainbow)
axs[5].set_ylim([freq_lims[0], freq_lims[1]])
axs[5].tick_params(axis='y', labelsize=18)
axs[5].set_xlim([xticks_horiz[0], xticks_horiz[-1]])
axs[5].set_xticks(xticks_horiz)
axs[5].set_xticklabels([], fontsize=22)
axs[5].set_yticks([0.5,5,10])
axs[5].set_yticklabels(['0.5','5','10'],fontsize=18)
axs[5].set_ylabel('PS4A.BDF\nFrequency',fontsize=22)

# Plot Infrasound ML classes
axs[6].imshow(matrix_plot_inf, cmap=cmap_inf, interpolation='None', extent=[xticks_horiz[0], xticks_horiz[-1], 0, 4], aspect='auto', origin='lower')
axs[6].imshow(1-matrix_alpha_inf, cmap=amap, interpolation='None', extent=[xticks_horiz[0], xticks_horiz[-1], 0, 4], aspect='auto', origin='lower')
axs[6].set_yticks(np.arange(0.5, 4.5, 1))
axs[6].set_yticklabels(np.flip(rgb_keys_inf))
axs[6].set_xticks(xticks_horiz)
axs[6].set_xticklabels([])
axs[6].set(xlabel=None)
axs[6].set_xlim([xticks_horiz[0], xticks_horiz[-1]])
axs[6].set_ylabel('Infrasound\nClass',fontsize=22)
axs[6].tick_params(axis='y', labelsize=18)

# Plot explosion times
axs[7].text(0,0.95,'ML ($\geq$0.6 $P_{norm}$ for both data types)',fontsize=18,color='black',horizontalalignment='left',verticalalignment='top')
axs[7].text(0,-0.05,'RTM (stack amplitude $\geq$0.9)',fontsize=18,color='red',horizontalalignment='left',verticalalignment='bottom')
axs[7].vlines(x=best_exp_pct_since, ymin=0.5, ymax=1, colors='black', ls='-', lw=0.5)
axs[7].vlines(x=rtm_pct_since, ymin=0, ymax=0.5, colors='red', ls='-', lw=0.5)
axs[7].set_ylim([0,1])
axs[7].set_yticklabels([])
axs[7].set_ylabel('Explosions',fontsize=22, rotation=0, labelpad=60, verticalalignment='center')
axs[7].set_xlim([xticks_horiz[0], xticks_horiz[-1]])
axs[7].set_xticks(xticks_horiz)
axs[7].set_xticklabels(xticklabels_horiz, fontsize=20, rotation=30)

# Save figure and show on interactive
fig.savefig(outfile, transparent=False, bbox_inches='tight')
fig.show()