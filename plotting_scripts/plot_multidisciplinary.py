# Plot multidisciplinary plot, top to bottom order
# 1. AVO color code
# 2. PS1A Spectrogram (Seismic)
# 2. Kernel Density Plot (Seismic)
# 3. Barcode Plot (Seismic)
# 4. Reduced displacement (Seismic)
# 5. Kernel Density Plot (Infrasound)
# 6. Barcode Plot (Infrasound)
# 7. Spaced-based observations (Hotspot & SO2)

# Import all dependencies
import pandas as pd
import numpy as np
import colorcet as cc
from obspy import UTCDateTime
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import medfilt

# Plot settings
common_norm=True
multiple='fill'
bw_method = 1/10000
outfile = '/Users/darrentpk/Desktop/multidisciplinary_fig_v4.png'

# Define time ticks on x-axis
base_time = UTCDateTime(2021,1,1)
total_days = (UTCDateTime(2023,3,1)-base_time)/86400
month_utcdatetimes = []
for i in range(1,13):
    month_utcdatetime = UTCDateTime(2021,i,1)
    month_utcdatetimes.append(month_utcdatetime)
for i in range(1,13):
    month_utcdatetime = UTCDateTime(2022,i,1)
    month_utcdatetimes.append(month_utcdatetime)
for i in range(1,4):
    month_utcdatetime = UTCDateTime(2023,i,1)
    month_utcdatetimes.append(month_utcdatetime)
xticks_horiz = [(t - base_time)/86400 for t in month_utcdatetimes]
xticklabels_horiz = [t.strftime('%b \'%y') for t in month_utcdatetimes]
xticks_horiz = [t/total_days for t in xticks_horiz]

# Define labeled start and end
labeled_start = UTCDateTime(2021, 7, 22)
labeled_end = UTCDateTime(2021, 9, 22)
labeled_start_pct_since = (UTCDateTime(2021, 7, 22)-base_time)/86400/total_days
labeled_end_pct_since = (UTCDateTime(2021, 9, 22)-base_time)/86400/total_days

# Define AVO color code axvspan limits
g2y_pct_since = (UTCDateTime(2021,7,9)-base_time)/86400/total_days
y2o_pct_since = (UTCDateTime(2021,8,5)-base_time)/86400/total_days
o2y_pct_since = (UTCDateTime(2022,12,17)-base_time)/86400/total_days
y2g_pct_since = (UTCDateTime(2023,1,19)-base_time)/86400/total_days
g2e_pct_since = (month_utcdatetimes[-1]-base_time)/86400/total_days

# Read spectrogram and reduced displacement data
metrics_dir = './metrics/PS1A_BHZ_metrics/'
tag = '_20210101_20230301'
tmpl = np.load(metrics_dir + 'tmpl_all' + tag + '.npy')
tmpl = tmpl[0,:]  # only need 1 row for time
dr = np.load(metrics_dir + 'dr_all' + tag + '.npy')
dr_nm = np.nanmedian(dr, axis=0)  # network median
dr_mf = medfilt(dr_nm, kernel_size=7)  # median filter
spec_db = np.load(metrics_dir + 'spec_db_all' + tag + '.npy')

# Read seismic dataframe with all properties
df = pd.read_csv ('./data_frames/seismic_df.csv', usecols=range(1,8))
df['class'] = df['class'].replace(np.nan,'N/A')

# Calculate days since 2021-01-01
date_list = list(df['time'])
pct_since_arr = np.array([(UTCDateTime(dt) - UTCDateTime(2021,1,1))/86400/total_days for dt in date_list])
df['pct_since'] = pct_since_arr

# Map classes into a continuous array for heatmap later
class_order = ['Broadband Tremor',
             'Harmonic Tremor',
             'Monochromatic Tremor',
             'Non-tremor Signal',
             'Explosion',
             'Noise',
             'N/A']
class_list = list(df['class'])
class_index = np.array([np.uint8(class_order.index(c)) for c in class_list])

# Define colors and hue order
rgb_values = np.array([
    [193,  39,  45],
    [  0, 129, 118],
    [  0,   0, 167],
    [238, 204,  22],
    [164,  98,   0],
    [ 40,  40,  40],
    [255, 255, 255]])
rgb_ratios = rgb_values/255
colors = {'Broadband Tremor': rgb_ratios[0],
          'Harmonic Tremor': rgb_ratios[1],
          'Monochromatic Tremor': rgb_ratios[2],
          'Non-tremor Signal': rgb_ratios[3],
          'Explosion': rgb_ratios[4],
          'Noise': rgb_ratios[5],
          'N/A': rgb_ratios[6]}
hue_order = ['Broadband Tremor',
             'Harmonic Tremor',
             'Monochromatic Tremor',
             'Non-tremor Signal',
             'Explosion',
             'Noise',
             'N/A']
listed_colors = np.concatenate((rgb_ratios, np.ones((np.shape(rgb_values)[0],1))), axis=1)
listed_colors = ListedColormap(listed_colors)
real_cbar_tick_interval = 2 * (7-1)/14
real_cbar_ticks = np.arange(real_cbar_tick_interval/2,7-1,real_cbar_tick_interval)

# Load thermal data
thermal_filepath = './data_frames/thermal_anomalies.csv'
thermal = pd.read_csv(thermal_filepath)
rp_time = list(thermal['image_time'])
rp_pct_since = np.array([(UTCDateTime(t)-base_time)/86400/total_days for t in rp_time])
nodata_pct_since = (UTCDateTime(2023,1,1)-base_time)/86400/total_days
text_pct_since = (UTCDateTime(2023,2,1)-base_time)/86400/total_days
rp_values = np.array(list(thermal['Radiative Power (MW)']))

# Load SO2 data
so2_filepath = './data_frames/so2_em_rate.csv'
so2 = pd.read_csv(so2_filepath)
so2_date = list(so2['date'])
so2_pct_since = np.array([((UTCDateTime(t)-base_time)/86400 + 0.5)/total_days  for t in so2_date])
so2_rates = np.array(list(so2['em_rate']))

# Read infrasound spectrogram
metrics_dir_inf = './metrics/PS4A_BDF_metrics/'
tmpl_inf = np.load(metrics_dir_inf + 'tmpl_all' + tag + '.npy')
tmpl_inf = tmpl_inf[0,:]  # only need 1 row for time
spec_db_inf = np.load(metrics_dir_inf + 'spec_db_all' + tag + '.npy')

# Read infrasound dataframe with all properties
df_inf = pd.read_csv ('./data_frames/infrasound_df.csv', usecols=range(1,8))
df_inf['class'] = df_inf['class'].replace(np.nan,'N/A')

# Calculate days since 2021-01-01
date_list_inf = list(df_inf['time'])
pct_since_arr_inf = np.array([(UTCDateTime(dt) - UTCDateTime(2021,1,1))/86400/total_days for dt in date_list_inf])
df_inf['pct_since'] = pct_since_arr_inf

# Map classes into a continuous array for heatmap later
class_order_inf = ['Infrasonic Tremor',
                  'Explosion',
                  'Wind Noise',
                  'Electronic Noise',
                  'N/A']
class_list_inf = list(df_inf['class'])
class_index_inf = np.array([np.uint8(class_order_inf.index(c)) for c in class_list_inf])

# Define colors and hue order
rgb_values_inf = np.array([
    [103,  52, 235],
    [235, 152,  52],
    [ 40,  40,  40],
    [ 15,  37,  60],
    [255, 255, 255]])
rgb_ratios_inf = rgb_values_inf/255
colors_inf = {'Infrasonic Tremor': rgb_ratios_inf[0],
              'Explosion': rgb_ratios_inf[1],
              'Wind Noise': rgb_ratios_inf[2],
              'Electronic Noise': rgb_ratios_inf[3],
              'N/A': rgb_ratios_inf[4]}
hue_order_inf = ['Infrasonic Tremor',
                 'Explosion',
                 'Wind Noise',
                 'Electronic Noise',
                 'N/A']
listed_colors_inf = np.concatenate((rgb_ratios_inf, np.ones((np.shape(rgb_values_inf)[0],1))), axis=1)
listed_colors_inf = ListedColormap(listed_colors_inf)
real_cbar_tick_interval_inf = 2 * (5-1)/10
real_cbar_ticks_inf = np.arange(real_cbar_tick_interval_inf/2,5-1,real_cbar_tick_interval_inf)

# Load explosion dataframe
exp_df = pd.read_csv('./data_frames/explosion_df.csv', usecols=range(1,7))
best_exp_df = exp_df[(exp_df['seismic_count']>=3) & (exp_df['infrasound_count']>=2)]
best_exp_times = [UTCDateTime(t) for t in best_exp_df['time'].tolist()]
best_exp_pct_since = [(t-base_time)/86400/total_days for t in best_exp_times]
rtm_df = pd.read_csv('./data_frames/pavlof_rtm.csv')
rtm_dates = rtm_df['Date'].tolist()
rtm_dists = rtm_df['Distance (M)'].tolist()
rtm_amps = rtm_df['Stack Amplitude'].tolist()
rtm_times = [UTCDateTime(rtm_dates[j]) for j in range(len(rtm_dates)) if ((rtm_dists[j] < 600) and rtm_amps[j]>0.9)]
rtm_pct_since = [(t-base_time)/86400/total_days for t in rtm_times]

# Commence multidisciplinary plotting
figsize = (37, 19)
height_ratios=[1.5,7,9,7,7,7,9,2]
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
sns.kdeplot(data=df[df['class']!='N/A'], x='pct_since', hue='class', hue_order=reversed(hue_order), fill=True,
        palette=colors, linewidth=1, legend=False, ax=axs[2], common_norm=common_norm, multiple=multiple, bw_method=bw_method)
axs[2].set_xticks(xticks_horiz)
axs[2].set_xticklabels([])
axs[2].set(xlabel=None)
axs[2].set_xlim([xticks_horiz[0], xticks_horiz[-1]])
ax2_ylim = axs[2].get_ylim()
new_ymin = ax2_ylim[1] * -0.2
axs[2].imshow(class_index.reshape(1,len(class_index)), extent=[0, 1, new_ymin, 0], origin='lower', aspect='auto',
              interpolation='None', cmap=listed_colors, alpha=0.75)
axs[2].axhline(y=0, linewidth=2.5, color='k')
axs[2].plot([labeled_start_pct_since, labeled_end_pct_since], [0.05*new_ymin, 0.05*new_ymin], 'k-', linewidth=5.5)
axs[2].plot([labeled_start_pct_since, labeled_end_pct_since], [0.95*new_ymin, 0.95*new_ymin], 'k-', linewidth=5.5)
axs[2].set_ylim([new_ymin, ax2_ylim[1]])
axs[2].set_ylabel('Seismic\nClass Density',fontsize=22)
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
sns.kdeplot(data=df_inf[df_inf['class']!='N/A'], x='pct_since', hue='class', hue_order=reversed(hue_order_inf), fill=True,
        palette=colors_inf, linewidth=1, legend=False, ax=axs[6], common_norm=common_norm, multiple=multiple, bw_method=bw_method)
axs[6].set_xticks(xticks_horiz)
axs[6].set_xticklabels([])
axs[6].set(xlabel=None)
axs[6].set_xlim([xticks_horiz[0], xticks_horiz[-1]])
ax6_ylim = axs[6].get_ylim()
new_ymin = ax6_ylim[1] * -0.2
axs[6].imshow(class_index_inf.reshape(1,len(class_index_inf)), extent=[0, 1, new_ymin, 0], origin='lower', aspect='auto',
              interpolation='None', cmap=listed_colors_inf, alpha=0.75)
axs[6].axhline(y=0, linewidth=2.5, color='k')
axs[6].plot([labeled_start_pct_since, labeled_end_pct_since], [0.05*new_ymin, 0.05*new_ymin], 'k-', linewidth=5.5)
axs[6].plot([labeled_start_pct_since, labeled_end_pct_since], [0.95*new_ymin, 0.95*new_ymin], 'k-', linewidth=5.5)
axs[6].set_ylim([new_ymin, ax6_ylim[1]])
axs[6].set_ylabel('Infrasound\nClass Density',fontsize=22)
axs[6].tick_params(axis='y', labelsize=18)

# Plot explosion times
axs[7].text(0,0.95,'ML ($\geq$3 seismic,$\geq$2 infrasound)',fontsize=18,color='black',horizontalalignment='left',verticalalignment='top')
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
fig.savefig(outfile, transparent=True)
fig.show()
