import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime

# Load seismic indicator (explosion is 4)
seismic_indicator = np.load('./classifications/seismic_indicators_unlabeled2.npy', allow_pickle=True)
seismic_df = pd.DataFrame(seismic_indicator, columns=['station','time','class'])
seismic_df = seismic_df[seismic_df['class']==4]
seismic_times = np.array(seismic_df['time'])

# Load infrasound indicator (explosion is 1)
infra_indicator = np.load('./classifications/infrasound_indicators_unlabeled2.npy', allow_pickle=True)
infra_df = pd.DataFrame(infra_indicator, columns=['station','time','class'])
infra_df = infra_df[infra_df['class']==1]
infra_times = np.array(infra_df['time'][infra_df['class']==1])

# All explosion times
all_times = np.concatenate((seismic_times, infra_times))
unique_times = np.sort(np.unique(all_times))

# seismic_lists = []
# seismic_counts = []
# infra_lists = []
# infra_counts = []
# total_counts = []
# for i, unique_time in enumerate(unique_times):
#     print('%d/%d' % (i,len(unique_times)))
#     seismic_list = list(seismic_df['station'][seismic_df['time'] == unique_time])
#     seismic_list = sorted(seismic_list)
#     infra_list = list(infra_df['station'][infra_df['time'] == unique_time])
#     infra_list = sorted(infra_list)
#     seismic_lists.append(seismic_list)
#     seismic_counts.append(len(seismic_list))
#     infra_lists.append(infra_list)
#     infra_counts.append(len(infra_list))
#     total_counts.append(len(seismic_list)+len(infra_list))
#
# # Create dataframe
# list_of_tuples = list(zip(list(unique_times), seismic_lists, seismic_counts, infra_lists, infra_counts, total_counts))
# exp_df = pd.DataFrame(list_of_tuples, columns = ['time',
#                                                  'seismic_stations',
#                                                  'seismic_count',
#                                                  'infrasound_stations',
#                                                  'infrasound_count',
#                                                  'total_count'])
# exp_df.to_csv('./explosion_df.csv')

# Load explosion dataframe
exp_df = pd.read_csv('./explosion_df2.csv', usecols=range(1,7))

# Load rtm times
rtm_df = pd.read_csv('./pavlof_rtm.csv')
rtm_dates = rtm_df['Date'].tolist()
rtm_dists = rtm_df['Distance (M)'].tolist()
rtm_amps = rtm_df['Stack Amplitude'].tolist()
rtm_times = [UTCDateTime(rtm_dates[j]) for j in range(len(rtm_dates)) if ((rtm_dists[j] < 600) and rtm_amps[j]>0.9)]

# Define fixed plot params
base_time = UTCDateTime(2021,1,1)
exp_days = [(UTCDateTime(t)-base_time)/86400 for t in exp_df['time']]
time_ticks_utc = []
for i in range(1,13):
    time_tick_utc = UTCDateTime(2021,i,1)
    time_ticks_utc.append(time_tick_utc)
for i in range(1,13):
    time_tick_utc = UTCDateTime(2022,i,1)
    time_ticks_utc.append(time_tick_utc)
time_ticks_utc.append(UTCDateTime(2023,1,1))
time_ticks = [(t-base_time)/86400 for t in time_ticks_utc]
time_ticklabels = [t.strftime('%b\'%y') for t in time_ticks_utc]

# Define variable plot params
width = 0.05
xmin = (UTCDateTime(2021,1,1)-base_time) / 86400
xmax = (UTCDateTime(2023,1,1)-base_time) / 86400
# xmin = (UTCDateTime(2022,7,1)-base_time) / 86400
# xmax = (UTCDateTime(2022,8,1)-base_time) / 86400

# # Do bar plot
# fig, ax = plt.subplots(figsize=(9,8))
# ax.bar(exp_days, exp_df['infrasound_count'], width, color='b')
# ax.bar(exp_days, exp_df['seismic_count'], width, bottom=exp_df['infrasound_count'], color='r')
# ax.set_ylabel('N(Stations)')
# ax.set_title('Explosion Classifications')
# ax.set_xticks(time_ticks, time_ticklabels, rotation=30)
# ax.set_yticks(np.arange(0, 10))
# ax.legend(labels=['Infrasound', 'Seismic'])
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([0, 10])
# fig.show()

# Look at high confidence explosion times?
best_exp_df = exp_df[(exp_df['seismic_count']>=3) & (exp_df['infrasound_count']>=2)]
best_exp_times = np.array([UTCDateTime(t) for t in best_exp_df['time'].tolist()])
best_exp_days = (best_exp_times-base_time)/86400

# map to index
station_list = ['PN7A','PS1A','PS4A','PV6A','PVV']
def map_to_index(name):
    return station_list.index(name)

# plot timeline of explosion returns
fig, axs = plt.subplots(2,1, figsize=(15,5))
for stn in station_list:
    station_seismic_times = seismic_df['time'][seismic_df['station'] == stn]
    station_seismic_days = (station_seismic_times - base_time) / 86400
    station_seismic_ind = np.ones(np.shape(station_seismic_days)) * station_list.index(stn) + 1
    station_infra_times = infra_df['time'][infra_df['station'] == stn]
    station_infra_days = (station_infra_times - base_time) / 86400
    station_infra_ind = np.ones(np.shape(station_infra_days)) * station_list.index(stn) + 1
    axs[0].scatter(station_seismic_days, station_seismic_ind, s=0.3, alpha=0.3)
    axs[1].scatter(station_infra_days, station_infra_ind, s=0.3, alpha=0.3)
axs[0].vlines(x=best_exp_days, ymin=0.5, ymax=5.5, colors='black', ls='-', lw=0.3)
axs[0].vlines(x=rtm_days, ymin=0.5, ymax=5.5, colors='red', ls='-', lw=0.1)
axs[1].vlines(x=best_exp_days, ymin=0.5, ymax=5.5, colors='black', ls='-', lw=0.3)
axs[1].vlines(x=rtm_days, ymin=0.5, ymax=5.5, colors='red', ls='-', lw=0.05)
axs[0].set_yticks(np.arange(1,6))
axs[0].set_yticklabels(station_list)
axs[0].set_ylim([0.5,5.5])
axs[0].set_ylabel('Seismic Stations')
axs[0].set_xlim([time_ticks[0],time_ticks[-1]])
axs[0].set_xticks([])
axs[0].invert_yaxis()
axs[1].set_yticks(np.arange(1,6))
axs[1].set_yticklabels(station_list)
axs[1].set_ylim([0.5,5.5])
axs[1].set_ylabel('Infrasound Stations')
axs[1].set_xlim([time_ticks[0],time_ticks[-1]])
axs[1].set_xticks(time_ticks, time_ticklabels, rotation=30)
axs[1].invert_yaxis()
fig.suptitle('Explosion classifications over time (with RTM)')
fig.subplots_adjust(hspace=0.05)
fig.show()





# Plot station-specific returns
station_list = ['PN7A','PS1A','PS4A','PV6A','PVV']
seismic_stations = exp_df['seismic_stations']
infrasound_stations = exp_df['infrasound_stations']

def map_to_bool(sublist):
    return [x in sublist for x in station_list]

seismic_bools = np.array(list(map(map_to_bool, seismic_stations))) + 0
seismic_bools = seismic_bools.transpose()
infrasound_bools = np.array(list(map(map_to_bool, infrasound_stations))) + 0
infrasound_bools = infrasound_bools.transpose()

fig, axs = plt.subplots(2, 1, figsize=(10,5))
axs[0].pcolor(seismic_bools, cmap='Greys')
axs[0].set_yticks(np.arange(0.5,5,1))
axs[0].set_yticklabels(station_list)
axs[0].invert_yaxis()
axs[0].set_ylabel('Seismic Stations')
axs[1].pcolor(infrasound_bools, cmap='Greys')
axs[1].set_yticks(np.arange(0.5,5,1))
axs[1].set_yticklabels(station_list)
axs[1].invert_yaxis()
axs[1].set_ylabel('Infrasound Stations')
axs[1].set_xlabel('Index of unique explosion time returns')
fig.show()



import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
# Load explosion dataframe
base_time = UTCDateTime(2021,1,1)
month_utcdatetimes = []
for i in range(1,13):
    month_utcdatetime = UTCDateTime(2021,i,1)
    month_utcdatetimes.append(month_utcdatetime)
for i in range(1,13):
    month_utcdatetime = UTCDateTime(2022,i,1)
    month_utcdatetimes.append(month_utcdatetime)
month_utcdatetimes.append(UTCDateTime(2023,1,1))
xticks_horiz = [(t - base_time)/86400 for t in month_utcdatetimes]
xticklabels_horiz = [t.strftime('%b \'%y') for t in month_utcdatetimes]
exp_df = pd.read_csv('./explosion_df.csv', usecols=range(1,7))
best_exp_df = exp_df[(exp_df['seismic_count']>=3) & (exp_df['infrasound_count']>=2)]
best_exp_times = [UTCDateTime(t) for t in best_exp_df['time'].tolist()]
best_exp_days_since = [(t-base_time)/86400 for t in best_exp_times]
rtm_df = pd.read_csv('./pavlof_rtm.csv')
rtm_dates = rtm_df['Date'].tolist()
rtm_dists = rtm_df['Distance (M)'].tolist()
rtm_amps = rtm_df['Stack Amplitude'].tolist()
rtm_times = [UTCDateTime(rtm_dates[j]) for j in range(len(rtm_dates)) if ((rtm_dists[j] < 600) and rtm_amps[j]>0.9)]
rtm_days_since = [(t-base_time)/86400 for t in rtm_times]
# Plot explosion times
fig, ax = plt.subplots(figsize=(10,1))
ax.text(0,0.95,'ML ($\geq$3 seismic, $\geq$2 infra)',fontsize=18,color='black',horizontalalignment='left',verticalalignment='top')
ax.text(-0.05,0,'RTM',fontsize=18,color='red',horizontalalignment='left',verticalalignment='bottom')
ax.vlines(x=best_exp_days_since, ymin=0.5, ymax=1, colors='black', ls='-', lw=0.5)
ax.vlines(x=rtm_days_since, ymin=0, ymax=0.5, colors='red', ls='-', lw=0.5)
ax.set_ylim([0,1])
ax.set_yticklabels([])
ax.set_ylabel('Explosions',fontsize=22, rotation=0, labelpad=60, verticalalignment='center')
ax.set_xlim([xticks_horiz[0], xticks_horiz[-1]])
ax.set_xticks(xticks_horiz)
ax.set_xticklabels(xticklabels_horiz, fontsize=20, rotation=30)
fig.show()
