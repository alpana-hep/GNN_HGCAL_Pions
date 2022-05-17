#import the stuff
import pandas as pd #dataframes etc
import matplotlib.pyplot as plt #plotting
import pickle
import numpy as np
import os, sys
import seaborn as sns

suma1 = np.load('./summaries_100.npz')
print('......')
# suma2 = np.load('./summaries_ep46.npz')

# print('....')
# suma3 = np.load('./summaries.npz')
# tr_loss1 = np.append(suma1['lr'],suma2['lr'])
# tr_loss = np.append(tr_loss1,suma3['lr'])

# va_loss1 = np.append(suma1['valid_loss'],suma2['valid_loss'])
tr_loss= suma1['lr']
va_loss = suma1['valid_loss']
# print(len(va_loss), len(tr_loss))
epochs = np.arange(len(va_loss))
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(epochs,tr_loss, label='learning rate')
#ax.plot(epochs,va_loss, label='validation loss')
ax.set_xlabel('Epochs',fontsize = 20)
ax.set_ylabel('learning rate',fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=30)
#ax.set_ylim(0.02,1.0)
fig.legend(loc=(0.7,0.8),prop={'size': 20})
#fig.show()
fig.savefig("lrvsepoch_5M_fixwt_ratio_dylr1e04_trimAHcal_2agg_NSM.png")


tr_loss= suma1['train_loss']
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(epochs,tr_loss, label='train_loss')
ax.plot(epochs,va_loss, label='validation loss')
ax.set_xlabel('Epochs',fontsize = 20)
ax.set_ylabel('Loss',fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=30)
fig.legend(loc=(0.7,0.8),prop={'size': 20})
fig.savefig("loss_vsepoch_5M_fixwt_ratio_dylr1e04_trimAHcal_2agg_NSM.png")
