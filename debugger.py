import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime

dir = '/Users/darrentpk/Desktop/aniakchakplot/'
repeater_file = dir + 'catalog.txt'
orphan_file = dir + 'orphancatalog.txt'
catalog_file = dir + 'earthquakes.csv'

rep_pd = pd.read_csv(repeater_file, names=['cluster','time'],header=None,index_col=None, delimiter=' ')
rep_t = rep_pd['time'].tolist()
repeaters = [UTCDateTime(t) for t in rep_t]

orp_pd = pd.read_csv(orphan_file, names=['time'],header=None,index_col=None, delimiter=' ')
orp_t = orp_pd['time'].tolist()
orphans = [UTCDateTime(t) for t in orp_t]

cat_pd = pd.read_csv(catalog_file)
cat_t = cat_pd['Date/Time UTC'].tolist()
catalog = [UTCDateTime(t) for t in cat_t]

base_time = UTCDateTime(2022,10,1)
end_time = UTCDateTime(2023,2,21)
day_vec = np.arange(base_time,end_time,86400)

rep_count = []
orp_count = []
both_count = []
cat_count = []
time_vec = []

for day in day_vec:
	time_vec.append((day-base_time)/86400)
	rep_count.append(np.sum([1 for rep in repeaters if rep<day]))
	orp_count.append(np.sum([1 for orp in orphans if orp<day]))
	both_count.append(rep_count[-1] + orp_count[-1])
	cat_count.append(np.sum([1 for cat in catalog if cat<day]))

xtick_utc = [UTCDateTime(2022,9,1),
			  UTCDateTime(2022,10,1),
			  UTCDateTime(2022,11,1),
			  UTCDateTime(2022,12,1),
			  UTCDateTime(2023,1,1),
			  UTCDateTime(2023,2,1)]
xtick_days = [(t - base_time)/86400 for t in xtick_utc]
xtick_labels = [t.strftime('%b \'%y') for t in xtick_utc]

plt.figure()
plt.step(time_vec,rep_count,'r-',label='Repeaters')
# plt.step(time_vec,orp_count,'b-',label='Orphans')
# plt.step(time_vec,both_count,color='purple',label='Combined')
plt.step(time_vec,cat_count,'k-',label='AVO catalog')
plt.legend()
plt.title('Aniakchak REDPy vs AVO catalog events, 2022-08-15 to 2023-02-21')
plt.ylabel('Number of events')
plt.xlabel('UTC Date')
plt.xticks(xtick_days,xtick_labels)
plt.xlim([0, (UTCDateTime(2023,2,15)-base_time)/86400])
plt.grid()
plt.show()

clusters = rep_pd['cluster'].tolist()
unique_clusters = list(set(clusters))
clust_count = []
for unique_cluster in unique_clusters:
	clust_count.append(clusters.count(unique_cluster))

plt.figure()
plt.hist(clust_count,bins=range(0,100,2),color='maroon',edgecolor='black')
plt.title('Distribution of cluster sizes')
plt.ylabel('No. of unique clusters')
plt.xlabel('No. of cluster members')
plt.xlim([0,60])
plt.ylim([0,200])
plt.grid()
plt.show()

