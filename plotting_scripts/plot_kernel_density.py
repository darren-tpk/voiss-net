# Plot kernel plot of properties pertaining to each class

# Import all dependencies
import pandas as pd
import numpy as np
from obspy import UTCDateTime
from seaborn import kdeplot
from matplotlib import pyplot as plt

# Read dataframe with all properties
df = pd.read_csv ('scatter_df.csv', usecols=range(1,8))
df['class'] = df['class'].replace(np.nan,'N/A')
df = df.loc[df['class'] != 'N/A']

# Define colors and hue order
rgb_values = np.array([
    [193,  39,  45],
    [  0, 129, 118],
    [  0,   0, 167],
    [238, 204,  22],
    [164,  98,   0],
    [ 40,  40,  40]])
rgb_ratios = rgb_values/255
colors = {'Broadband Tremor': rgb_ratios[0],
          'Harmonic Tremor': rgb_ratios[1],
          'Monochromatic Tremor': rgb_ratios[2],
          'Non-tremor Signal': rgb_ratios[3],
          'Explosion': rgb_ratios[4],
          'Noise': rgb_ratios[5]}
hue_order = ['Broadband Tremor',
             'Harmonic Tremor',
             'Monochromatic Tremor',
             'Non-tremor Signal',
             'Explosion',
             'Noise']

# Define time ticks
month_utcdatetimes = []
for i in range(1,13):
    month_utcdatetime = UTCDateTime(2021,i,1)
    month_utcdatetimes.append(month_utcdatetime)
for i in range(1,13):
    month_utcdatetime = UTCDateTime(2022,i,1)
    month_utcdatetimes.append(month_utcdatetime)
month_utcdatetimes.append(UTCDateTime(2023,1,1))
xticks_horiz = [(t - UTCDateTime(2021,1,1)) for t in month_utcdatetimes]
xticklabels_horiz = [t.strftime('%b \'%y') for t in month_utcdatetimes]
xticks_horiz = [(t/86400)/730 for t in xticks_horiz]

# Calculate days since 2021-01-01
date_list = list(df['time'])
pct_since_arr = np.array([(UTCDateTime(dt) - UTCDateTime(2021,1,1))/(86400*730) for dt in date_list])
df['pct_since'] = pct_since_arr

# Plot kernel plot
fig, ax = plt.subplots(figsize=(10,2))
kdeplot(data=df, x='pct_since', hue='class', hue_order=hue_order, fill=True, bw_method=4/len(df),
        palette=colors, linewidth=1, legend=False, ax=ax, common_norm=True)
ax.set_xticks(xticks_horiz)
ax.set_xticklabels(xticklabels_horiz, rotation=30)
ax.set_xlim([0,1])
fig.show()

# Plot kernel plot
fig, ax = plt.subplots(figsize=(20,2))
kdeplot(data=df, x='pct_since', hue='class', hue_order=reversed(hue_order), fill=True, bw_method=len(hue_order)/(2*len(df)),
        palette=colors, linewidth=0, legend=False, ax=ax, common_norm=True, multiple='fill')
ax.set_xticks(xticks_horiz)
ax.set_xticklabels(xticklabels_horiz, rotation=30)
ax.set_xlim([0,1])
fig.show()

fig, ax = plt.subplots(figsize=(20,2))
for c in hue_order:
    kdeplot(data=df[df['class']==c], x='pct_since', fill=True, bw_method=2/(len(df[df['class']==c])),
            palette=colors, linewidth=0, legend=False, ax=ax)
ax.set_xticks(xticks_horiz)
ax.set_xticklabels(xticklabels_horiz, rotation=30)
ax.set_xlim([0,1])
fig.show()






import seaborn as sns
x1 = np.arange(1/30,17/30,1/15)
x2 = np.arange(23/30,1,1/15)
x = np.concatenate((x1,x2))
y = np.arange(17/30,22/30,1/15)
xy = np.concatenate((x,y))
xc = ['BB' for i in range(len(x))]
yc = ['MT' for i in range(len(y))]
xyc = [*xc, *yc]

# wts1 = [len(x)/len(y) for i in range(len(x))]
# wts2 = [len(y)/len(x) for i in range(len(y))]
# wts = [*wts1, *wts2]

df = pd.DataFrame(list(zip(xy,xyc)),columns=['val','class'])
fig, ax = plt.subplots()
sns.kdeplot(data=df, x='val', hue='class', bw_method=2/(len(xy)),
            common_norm=True, ax=ax)
# t = sns.kdeplot(x, bw_method=2/(len(x)), ax=ax)
# t = sns.kdeplot(y, bw_method=2/(len(y)), weights=[], ax=ax)
fig.show()
