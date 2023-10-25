### Calculate uncertainty from model training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

seis_filepath = '/Users/darrentpk/Desktop/GitHub/tremor_ml/seis_metrics.txt'
infra_filepath = '/Users/darrentpk/Desktop/GitHub/tremor_ml/infra_metrics.txt'

seis_df = pd.read_csv(seis_filepath,names=['sd','acc','pre','rec','f1'],header=0)
infra_df = pd.read_csv(infra_filepath,names=['sd','acc','pre','rec','f1'],header=0)

print(seis_df['acc'].mean()*100, 200*seis_df['acc'].std())
print(infra_df['acc'].mean()*100, 200*infra_df['acc'].std())

fig, ax = plt.subplots(2,1, figsize=(8,8))
ax[0].hist(seis_df['acc'],bins=np.arange(0.74,0.86,0.005),edgecolor='grey',color='lightgrey')
mean_label = 'Mean (%.3f)' % seis_df['acc'].mean()
model_label = 'Model (%.3f)' % 0.810
ax[0].axvline(x=seis_df['acc'].mean(),color='k',linestyle='--',linewidth=4,alpha=.6,label=mean_label)
ax[0].axvline(x=0.810,color='r',linestyle='--',linewidth=4,alpha=.6,label=model_label)
ax[0].set_title('Test accuracy distribution of 50 seismic CNNs',fontsize=20)
ax[0].set_xlim([0.74,0.86])
ax[0].set_xlabel('Test Accuracy',fontsize=18)
ax[0].set_yticks(np.arange(0,14,2))
ax[0].legend(fontsize=16)
ax[0].tick_params(axis='both', labelsize=16)
ax[1].hist(infra_df['acc'],bins=np.arange(0.82,0.94,0.005),edgecolor='grey',color='lightgrey')
mean_label = 'Mean (%.3f)' % infra_df['acc'].mean()
model_label = 'Model (%.3f)' % 0.889
ax[1].axvline(x=infra_df['acc'].mean(),color='k',linestyle='--',linewidth=4,alpha=.6,label=mean_label)
ax[1].axvline(x=0.889,color='r',linestyle='--',linewidth=4,alpha=.6,label=model_label)
ax[1].set_title('Test accuracy distribution of 50 infrasound CNNs',fontsize=20)
ax[1].set_xlim([0.82,0.94])
ax[1].set_xlabel('Test Accuracy',fontsize=18)
ax[1].set_yticks(np.arange(0,14,2))
ax[1].tick_params(axis='both', labelsize=16)
ax[1].legend(fontsize=16,loc='upper left')
fig.tight_layout(pad=2.5)
fig.show()
# fig.savefig('/Users/darrentpk/Desktop/simulations.png',bbox_inches='tight',transparent=False)