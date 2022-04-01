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
import matplotlib.cm as mcm
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


cmap = mcm.get_cmap('inferno')
colors = []
colors.append(cmap(0.2+1/3))
colors.append(cmap(0.2+0/3))
colors.append(cmap(0.2+2/3))

i = 0.01 # kappaI
pa = 0.1 # mean prey per grid cell at time=0

fs = 25

n_bins = 10
x_bins = np.arange(n_bins)

def make_subplot(ax, con='RW', t=0.01,
                 title = '', nfi = 5, perc = 75):
    symb = ['^','s', 'D', 'o']
    # , f, kp, p
    FADp = {'present (%)':np.array([]),
            'days':np.array([]),
            'Behaviour':np.array([]),
            'Strategy':np.array([])
            }
    for p in [0.95, 0, -1, -2]:
        for f in [0.5, 1, 1.5]:
            for kp in [0.5, 1., 1.5]:
            
                dirr = 'input/'
                nc = Dataset(dirr+'FADpa%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa))
    
                nf, nt = (nc['FADs'][:], nc['tuna'][:])#np.meshgrid
                for it in range(1):
                    for nti in range(len(nt)):
                        FADpc = nc['Fa'][nfi,nti,0,:nf[nfi]]
                        for ind in range(FADpc.shape[0]):
                            FADp['present (%)'] = np.append(FADp['present (%)'],
                                                            FADpc[ind])
                            FADp['days'] = np.append(FADp['days'],
                                                     np.arange(FADpc.shape[1])/2)
                            if(p==-1):
                                FADp['Strategy'] = np.append(FADp['Strategy'],
                                                            np.full(len(FADpc[ind]),
                                                                    'FS1'))
                            elif(p==0):
                                FADp['Strategy'] = np.append(FADp['Strategy'],
                                                            np.full(len(FADpc[ind]),
                                                                    'FS2'))
                            elif(p==-2):
                                FADp['Strategy'] = np.append(FADp['Strategy'],
                                                            np.full(len(FADpc[ind]),
                                                                    'FS0'))
                            else:
                                FADp['Strategy'] = np.append(FADp['Strategy'],
                                                            np.full(len(FADpc[ind]),
                                                                    'FS3'))
                            if(f>kp):
                                FADp['Behaviour'] = np.append(FADp['Behaviour'],
                                                            np.full(len(FADpc[ind]),
                                                                    r'$\kappa^F>\kappa^P$'))
                            elif(f==kp):
                                FADp['Behaviour'] = np.append(FADp['Behaviour'],
                                                            np.full(len(FADpc[ind]),
                                                                    r'$\kappa^F=\kappa^P$'))
                            else:
                                FADp['Behaviour'] = np.append(FADp['Behaviour'],
                                                            np.full(len(FADpc[ind]),
                                                                    r'$\kappa^F<\kappa^P$'))
    dfp = pd.DataFrame.from_dict(FADp)
    labels = np.arange(0,105,5)
    labelsc = labels[:-1]
    dfp['day bin'] = pd.cut(dfp['days'], bins=labels,
                            labels=labelsc+2.5)
    plt.title(title, fontsize=fs)
#    if(title[:3]!='(e)'):
    if(True):
        legend = False
        g = sns.lineplot(x="day bin",  y="present (%)", data=dfp,
                 ax=ax,
                 hue='Behaviour',
                 style='Strategy',
                 markers=symb,#True,
                 dashes=False, palette=colors,
                 err_kws={'alpha':0},
                 legend=legend)
    else:
        legend = 'full'
        sns.lineplot(x="day bin",  y="present (%)", data=dfp,
                 ax=ax,
                 hue='Behaviour',
                 style='Strategy',
                 markers=symb,#True,
                 dashes=False, palette=colors,
                 err_kws={'alpha':0},
                 legend=legend)
        ax.legend(bbox_to_anchor=(2.45, 0.9), borderaxespad=0., fontsize=fs-10)
    for child in ax.get_children():
        try:
            child.set_markeredgecolor('k')
            child.set_path_effects([pe.Stroke(linewidth=3, foreground='k'),
                                    pe.Normal()])
        except:
            pass
    ax.set_title(title, fontsize=fs)
    if(title[:3] in ['(a)','(c)']):
        ax.set(yscale='log')
    ax.set_ylabel('')
    ax.set_xlim(0,100)
    if(title[:3]!='(e)'):
        ax.set_xticklabels([])
        ax.set_xlabel('')
    ax.set_ylim(bottom=0)


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

def create_dict(nc, FADpd, nf, nfi, days, beha='', strat=''):
    FADp = nc['Fa'][nfi,0,0,:nf[nfi]]
    Fe =  nc['Fc'][nfi,0,0,1:nf[nfi]+1]
    be, ae, fe = Fishing_events(Fe)
    for i in range(len(days)):
        if(days[i]!=0):
            res = FADp[be==days[i]].flatten()
            FADpd['present (%)'] = np.append(FADpd['present (%)'],
                                             res)
            FADpd['days'] = np.append(FADpd['days'],
                                             np.full(res.shape, days[i]))
            FADpd['Strategy'] = np.append(FADpd['Strategy'],
                                        np.full(res.shape,
                                                strat))
            FADpd['Behaviour'] = np.append(FADpd['Behaviour'],
                                        np.full(res.shape,
                                                beha))
            
            res = FADp[ae==days[i]].flatten()
            FADpd['present (%)'] = np.append(FADpd['present (%)'],
                                             res)
            FADpd['days'] = np.append(FADpd['days'],
                                             np.full(res.shape, days[i]))
            FADpd['Strategy'] = np.append(FADpd['Strategy'],
                                        np.full(res.shape,
                                                strat))
            FADpd['Behaviour'] = np.append(FADpd['Behaviour'],
                                        np.full(res.shape,
                                                beha))
        else:
            res = FADp[fe==1].flatten()
            FADpd['present (%)'] = np.append(FADpd['present (%)'],
                                             res)
            FADpd['days'] = np.append(FADpd['days'],
                                             np.zeros(res.shape))
            FADpd['Strategy'] = np.append(FADpd['Strategy'],
                                        np.full(res.shape,
                                                strat))
            FADpd['Behaviour'] = np.append(FADpd['Behaviour'],
                                        np.full(res.shape,
                                                beha))
    return FADpd


def make_subplot_fe(ax, f=0.5, kp=0.5, p=0, con='RW', t=0.01,
                 title = '', days = np.arange(-11,11,0.5), nfi=5):
    symb = ['^','s', 'D', 'o']
    dirr = 'input/'
    nc = Dataset(dirr+'FADpa%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa))
    
    nf, nt = (nc['FADs'][:], nc['tuna'][:])#np.meshgrid
    print('number of dFADs:',nf[nfi])
    for it in range(1):
        for nti in range(len(nt)):
            FADp = {'present (%)':np.array([]),
                  'days':np.array([]),
                  'Behaviour':np.array([]),
                  'Strategy':np.array([])
                  }
            
            for p in [0.95, 0, -1]:
                for f in [0.5, 1, 1.5]:
                    for kp in [0.5, 1., 1.5]:
                        nc = Dataset(dirr+'FADpa%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa))
                        if(p==-1):
                            strat= 'FS1'
                        elif(p==0):
                            strat= 'FS2'
                        else:
                            strat= 'FS3'
                        if(f==kp):
                            beha = r'$\kappa^F=\kappa^P$'
                        elif(f>kp):
                            beha = r'$\kappa^F>\kappa^P$'
                        else:
                            beha = r'$\kappa^F<\kappa^P$'
                        FADp = create_dict(nc, FADp, nf,
                                          nfi, days, strat=strat, beha=beha)
    dfp = pd.DataFrame.from_dict(FADp)
    
    labels = [(i-0.5) for i in np.unique(dfp['days'])[::2]]
    labels = np.sort(np.unique(labels))
    labelsc = ['%d'%(i-0.5) for i in labels[1:]]
    dfp['day bin'] = pd.cut(dfp['days'], bins=labels,
                   labels=labelsc)
    sns.lineplot(x="day bin",  y="present (%)", data=dfp,
                 ax=ax,
                 hue='Behaviour',
                 style='Strategy',
                 markers=symb,
                 dashes=False, palette=colors,
                 err_kws={'alpha':0},
                 legend=False)
    for child in ax.get_children():
        try:
            child.set_markeredgecolor('k')
            child.set_path_effects([pe.Stroke(linewidth=5, foreground='k'),
                                    pe.Normal()])
        except:
            pass
        
    ax.set_title(title,
              fontsize=fs)
    ax.set_ylabel('')
    if(title[:3] in ['(b)','(d)']):
        ax.set(yscale='log')
    if(title[:3]!='(f)'):
        ax.set_xticklabels([])
        ax.set_xlabel('')

def set_label(ax, fs=fs):
    ax.set_xlabel('')
    ax.set_ylabel('number of associated tuna', fontsize=fs)
def set_label2(ax, fs=fs):
    plt.setp(ax.get_xticklabels(), fontsize=fs-5)
    plt.setp(ax.get_yticklabels(), fontsize=fs-5)


fig, ax = plt.subplots(3,3, figsize=(19,14),
                       gridspec_kw={'height_ratios': [10, 10,10],
                                    'width_ratios': [20, 20, 3]},
 #                      sharex = 'col',
                       )
# Set parameters for plotting
nfi = 1
t = 0 # 0.01

#%% days after deployment
make_subplot(ax[0,0], con='RW',
             title='(a) Random walk', nfi=nfi, t=t)  
make_subplot(ax[1,0], con='DG',
              title='(c) Double Eddy', nfi=nfi, t=t)  
make_subplot(ax[2,0], con='BJ',
              title='(e) Bickley Jet', nfi=nfi, t=t)

set_label(ax[1,0])
set_label2(ax[0,0])
set_label2(ax[0,1])
set_label2(ax[1,0])
set_label2(ax[1,1])
set_label2(ax[2,0])
set_label2(ax[2,1])
#%% days after fishing event
# 'do not use p==-1, then no fishing events take place at FADs'
make_subplot_fe(ax[0,1], con='RW',
              title='(b) Random Walk', nfi=nfi, t=t)
make_subplot_fe(ax[1,1], con='DG',
               title='(d) Double Eddy', nfi=nfi, t=t)
make_subplot_fe(ax[2,1], con='BJ',
               title='(f) Bickley Jet', nfi=nfi, t=t)

ax[0,2].remove()
ax[1,2].remove()
ax[2,2].remove()

# legends
custom_lines = []
for fi in range(3):
    custom_lines.append(Line2D([0], [0], color=cmap(0.2+fi/3),
                               lw=2, marker='o', markeredgecolor='k',
                               path_effects=[pe.Stroke(linewidth=5, foreground='k'),
                                    pe.Normal()]))
le1 = ax[1,1].legend(custom_lines, [r'$\kappa^P>\kappa^F$',
                                    r'$\kappa^P=\kappa^F$',
                                    r'$\kappa^P<\kappa^F$'],
                     title='Behaviour',
             loc='upper left', bbox_to_anchor=(1.02, 1.2),
             fontsize=fs-3)

frame = le1.get_frame()
frame.set_facecolor('lightgray')
plt.setp(le1.get_title(),fontsize=fs-10)

custom_lines = []
symb = ['o','D','s','^']
for fi in range(4):
    custom_lines.append(Line2D([0], [0], color='k',
                               lw=1, marker=symb[fi], markeredgecolor='k',
                               markersize=10,
                               linewidth=2
   #                            path_effects=[pe.Stroke(linewidth=2,
   #                                                    foreground='k'),
   #                                 pe.Normal()]
                               ))
le1 = ax[2,1].legend(custom_lines, ['FS0',
                                    'FS1',
                                    'FS2',
                                    'FS3'],
                     title='Strategy',
             loc='lower left', bbox_to_anchor=(1.02, 0.8),
             fontsize=fs-3)

frame = le1.get_frame()
frame.set_facecolor('lightgray')

plt.setp(le1.get_title(),fontsize=fs-10)


ax[-1,0].set_xlabel('Drifting time after deployment (days)', fontsize=fs)
ax[-1,1].set_xlabel('Time after set (days)', fontsize=fs)

#fig.tight_layout()
plt.savefig('figure2_t%.2f_nfi%d.pdf'%(t,nfi), bbox_inches='tight')
plt.show()






