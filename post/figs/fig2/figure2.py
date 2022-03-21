#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:44:48 2021

@author: nooteboom
"""
import numpy as np
from netCDF4 import Dataset
import matplotlib.pylab as plt
import seaborn as sns
sns.set()
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

i = 0.01 # kappaI
pa = 0.1 # mean prey per grid cell at time=0

fs = 25

n_bins = 10
x_bins = np.arange(n_bins)

def make_subplot(ax, f, kp, p, con='RW', t=0.01,
                 title = '', nfi = 5):
    dirr = 'input/'
    nc = Dataset(dirr+'FADpa%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa))
    
    nf, nt = (nc['FADs'][:], nc['tuna'][:])#np.meshgrid
    for it in range(1):
        for nti in range(len(nt)):
            nof = 0
            FADpc = nc['Fa'][nfi,nti,0,:nf[nfi]]
            for ind in range(FADpc.shape[0]):
                if(nof==0):
                    FADp = {'present (%)':FADpc[0],
                          'days':np.arange(FADpc.shape[1])/2
                          }
                else:
                    FADp['present (%)'] = np.append(FADp['present (%)'],
                                                     FADpc[ind])
                    FADp['days'] = np.append(FADp['days'],
                                             np.arange(FADpc.shape[1])/2)
                nof += 1
            dfp = pd.DataFrame.from_dict(FADp)
            labels = np.arange(0,105,5)
            labelsc = labels[:-1]
            dfp['day bin'] = pd.cut(dfp['days'], bins=labels,
                           labels=labelsc)
            plt.title(title, fontsize=fs)
            sns.boxplot(x="day bin",  y="present (%)", data=dfp,
                              color='cornflowerblue',
                              medianprops=dict(color="k", alpha=0.9),
                              ax=ax, whis=[5, 95])
            ax.set_title(title, fontsize=fs)

def create_dict(FADp, be,ae, fe, nf, nfi, days):
    FADpd = {'present (%)':np.array([]),
          'days':np.array([])
          }
    for i in range(len(days)):
        if(days[i]!=0):
            res = FADp[be==days[i]].flatten()
            FADpd['present (%)'] = np.append(FADpd['present (%)'],
                                             res)
            FADpd['days'] = np.append(FADpd['days'],
                                             np.full(res.shape, days[i]))
            res = FADp[ae==days[i]].flatten()
            FADpd['present (%)'] = np.append(FADpd['present (%)'],
                                             res)
            FADpd['days'] = np.append(FADpd['days'],
                                             np.full(res.shape, days[i]))
        else:
            res = FADp[fe==1].flatten()
            FADpd['present (%)'] = np.append(FADpd['present (%)'],
                                             res)
            FADpd['days'] = np.append(FADpd['days'],
                                             np.zeros(res.shape))   
    return FADpd

def Fishing_events(Fe):
    fishevents = np.zeros(Fe.shape)
    afterevent = np.zeros(Fe.shape)
    beforeevent = np.zeros(Fe.shape)
    for fi in range(Fe.shape[0]):
        for ti in range(1,Fe.shape[1]-1):
            if(Fe[fi,ti]!=Fe[fi,ti-1]):
                fishevents[fi,ti] = 1
    for fi in range(Fe.shape[0]):
        bo = False
        bo2 = False
        co2 = 0
        for ti in range(Fe.shape[1]):
            if(fishevents[fi,ti]==1):
                bo = True
                co = 0
            elif(bo):
                co += 0.5
                afterevent[fi,ti] = co
            if(fishevents[fi,-ti]==1):
                bo2 = True
                co2 = 0
            elif(bo2):
                co2 -= 0.5
                beforeevent[fi,-ti] = co2
    return beforeevent, afterevent, fishevents

def make_subplot_fe(ax, f, kp, p, con='RW', t=0.01,
                 title = '', days = np.arange(-11,11,0.5), nfi=5):
    dirr = 'input/'
    nc = Dataset(dirr+'FADpa%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa))

    nfc = 0
    ntc = 0
    
    nf, nt = (nc['FADs'][nfc:], nc['tuna'][ntc:])#np.meshgrid
    print('number of dFADs:',nf[nfi])
    for it in range(1):
        for nti in range(len(nt)):
            FADp = {'present (%)':np.array([]),
                  'days':np.array([])
                  }
            FADp = nc['Fa'][nfc+nfi,ntc+nti,0,:nf[nfi]]
            Fe =  nc['Fc'][nfc+nfi,ntc+nti,0,1:nf[nfi]+1]
            be, ae, fe = Fishing_events(Fe)
            FADp = create_dict(FADp, be,ae, fe, nf,
                              nfi, days)

            dfp = pd.DataFrame.from_dict(FADp)
            
            labels = [(i-0.5) for i in np.unique(dfp['days'])[::2]]
            labels = np.sort(np.unique(labels))
            labelsc = ['%d'%(i-0.5) for i in labels[1:]]
            dfp['day bin'] = pd.cut(dfp['days'], bins=labels,
                           labels=labelsc)
            sns.boxplot(x="day bin",  y="present (%)", data=dfp,
                        color='cornflowerblue',
                        medianprops=dict(color="k", alpha=0.9),
                        ax=ax, whis=[5, 95],
                        showfliers = False)
                
            ax.set_title(title,
                      fontsize=fs)
            ax.set_ylabel('')

def set_label(ax, fs=fs):
    ax.set_xlabel('')
    ax.set_ylabel('number of associated tuna', fontsize=fs)
def set_label2(ax, fs=fs):
    plt.setp(ax.get_xticklabels(), fontsize=fs-5)
    plt.setp(ax.get_yticklabels(), fontsize=fs-5)


fig, ax = plt.subplots(3,3, figsize=(19,16),
                       gridspec_kw={'height_ratios': [12, 10, 10],
                                    'width_ratios': [20, 20, 3]}
                       )
# Set parameters for plotting
f = 1
kp = 1
nfi = 3
#%% days after deployment
p = -2
make_subplot(ax[1,0], f=f, kp=kp, p=p, con='RW',
             title='(c) Random walk, FS0', nfi=nfi)  
make_subplot(ax[2,0], f=f, kp=kp, p=p, con='BJ',
             title='(e) Bickley Jet, FS0', nfi=nfi) 

set_label(ax[1,0])
set_label(ax[2,0])
set_label2(ax[1,0])
set_label2(ax[1,1])
set_label2(ax[2,0])
set_label2(ax[2,1])
#%% days after fishing event
# 'do not use p==-1, then no fishing events take place at FADs'
make_subplot_fe(ax[1,1], f=f, kp=kp, p=0, con='BJ',
             title='(d) Bickley Jet, FS2', nfi=nfi) 
ax[1,1].set_xlabel('')
make_subplot_fe(ax[2,1], f=f, kp=kp, p=0.95, con='BJ',
             title='(f) Bickley Jet, FS3', nfi=nfi)  
#%% Observations

img = mpimg.imread('tfigs/clusFADT.png')
ax[0,0].imshow(img)
ax[0,0].set_title('(a) Observations', fontsize=fs)
ax[0,0].set_ylabel('percentage of cluster', fontsize=fs)
ax[0,0].get_xaxis().set_ticks([])
ax[0,0].get_yaxis().set_ticks([])
img = mpimg.imread('tfigs/clusFAD_fishingevent.png')
ax[0,1].imshow(img)
ax[0,1].set_title('(b) Observations', fontsize=fs)
ax[0,1].get_xaxis().set_ticks([])
ax[0,1].get_yaxis().set_ticks([])
img = mpimg.imread('tfigs/legend.png')
ax[0,2].imshow(img)
ax[0,2].get_xaxis().set_ticks([])
ax[0,2].get_yaxis().set_ticks([])
#ax[0,2].set_position(pos = [0.6,0.25,0.05,0.05], which='both')

ax[1,2].remove()
ax[2,2].remove()

ax[-1,0].set_xlabel('Drifting time after deployment (days)', fontsize=fs)
ax[-1,1].set_xlabel('Time after set (days)', fontsize=fs)

fig.tight_layout()
plt.savefig('figure2.pdf', bbox_inches='tight')
plt.show()






