## Workspace for debugging

from libcomcat.search import search
from obspy import UTCDateTime

t1=UTCDateTime(2021,7,1)
t2=UTCDateTime(2021,8,30)
earthquakes = search(starttime=t1.datetime,
					 endtime=t2.datetime,
					 latitude=55.417,
    			 	 longitude=-161.894,
    			     maxradiuskm=275,
					 reviewstatus='reviewed')

print('Found {:g} earthquakes.'.format(len(earthquakes)))

for eq in earthquakes:
	if eq.magnitude == 8.2:
		print(eq)

