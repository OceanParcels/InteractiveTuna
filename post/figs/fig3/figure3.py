#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:24:22 2021

@author: nooteboom
"""
import numpy as np
from netCDF4 import Dataset
import matplotlib.pylab as plt
import seaborn as sns
sns.set()
import matplotlib.cm as mcm
from matplotlib.lines import Line2D

from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size(13)

cmap = mcm.get_cmap('inferno')

fa = [0.,0.5, 1.,1.5]# kappaF
kpa = [0.,0.5, 1.,1.5]# kappaP

fs = 25

def plot_obs(ax):
    cos = ['k', 'tab:red', 'tab:blue']
    # define quantiles
    dic = {}

    # mean values
    dic['BETFm'] = 0.0226173391060002 * 100
    dic['BETfm'] = 0.205482266219817 * 100
    dic['SKJFm'] = 0.054317799601853 * 100
    dic['SKJfm'] = 0.19608406163893 * 100
    dic['YFTFm'] = 0.158040661442371 * 100
    dic['YFTfm'] = 0.314216508889378 * 100

    ax.scatter(dic['BETfm'],dic['BETFm'],marker = 'P', c=cos[0], s=190, edgecolor='k')
    ax.scatter(dic['SKJfm'],dic['SKJFm'],marker = 'P', c=cos[1], s=190, edgecolor='k')
    ax.scatter(dic['YFTfm'],dic['YFTFm'],marker = 'P', c=cos[2], s=190, edgecolor='k')

def fStomach2(nc, nf, nfi, nt, nti, nfc=0, ntc=0, it=0):
    nct = nc['Tsta'][nfc+nfi,ntc+nti,it,nf[nfi]:]
    nct2 = nc['Tstna'][nfc+nfi,ntc+nti,it,nf[nfi]:]

    if(not nct.mask.all()):
        Sta = np.nanmean(nct)
    else:
        Sta = np.nan
    if(not nct2.mask.all()):
        Stna = np.nanmean(nct2)
    else:
        Stna = np.nan
    return Sta, Stna

def make_subplot(ax, con='BJ', i=0.01, nfi = 5, pa=0.1,
                 title='', leg=True):
    symb = ['o','D','s','^']
    ax.plot([0,100],[0,100], '--', linewidth=2, c='k', alpha=0.8)
    dirr = 'input/'    
    
    for pi, p in enumerate([-2,-1, 0, 0.95]):
        for ti, t in enumerate([0, 0.01]):
            Sta = np.zeros((len(fa), len(kpa)))
            Stna = np.zeros((len(fa), len(kpa)))
            for fi, f in enumerate(fa):
                for kpi, kp in enumerate(kpa):
                    nc = Dataset(dirr + 'FADst%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa))
                    nf, nt = (nc['FADs'][:], nc['tuna'][:])#np.meshgrid
                    print('nf: ',nf[nfi])
                    for it in range(1):
                        for nti in range(len(nt)):
                            Sta[fi,kpi], Stna[fi,kpi] = fStomach2(nc, nf, nfi,
                                                                  nt, nti) 
            # from index to percentage
            Sta *= 100
            Stna *= 100   
        
            assert Sta.shape[0]==Sta.shape[1]
            assert Stna.shape[0]==Sta.shape[1]
            n, m = Sta[1:,1:].shape
            m = np.tril_indices(n=n, k=-1, m=m)
            Sta = np.array([
                           np.mean(Sta[0,1:]),
                            np.mean(np.transpose(Sta)[m]),
                            np.trace(Sta[1:,1:])/n,
                           np.mean(Sta[1:,1:][m]),
                           np.mean(Sta[1:,0])])
            Stna = np.array([
                           np.mean(Stna[0,1:]),
                            np.mean(np.transpose(Stna[1:,1:])[m]),
                            np.trace(Stna[1:,1:])/n,
                           np.mean(Stna[1:,1:][m]),
                           np.mean(Stna[1:,0])])
            
            color = [cmap((i+1)/5) for i in range(5)]
            if(t==0):
                ax.scatter(Stna, Sta, s = size, marker=symb[pi],
                       color=color, alpha=0.7, edgecolor='k',
                       linewidth=4)
            else:
                ax.scatter(Stna, Sta, s = size, marker=symb[pi],
                       color=color, alpha=0.7, edgecolor='k',
                       linewidth=1)#, linestyle='dotted')

    ax.set_title(title, fontsize=fs)
    plot_obs(ax)
    
    if(con=='DG' and nfi==5):
        ax.set_ylim(0,44)
        ax.set_xlim(0,44)
    else:
        ax.set_ylim(0,50)
        ax.set_xlim(0,50)
    if(con=='BJ' and nfi==5):
        ax.set_ylim(0,65)
        ax.set_xlim(0,65)
    elif(con=='BJ'):
        ax.set_ylim(0,70)
        ax.set_xlim(0,70)
    
    if(leg):
        #legend
        custom_lines = []
        for fi in range(5):
            custom_lines.append(Line2D([0], [0], color=cmap((1+fi)/5),
                                lw=0, marker='o', markeredgecolor='k',
                                markersize=8, alpha=0.7))
        names = [
                 r'$\kappa^P>0$ and $\kappa^F=0$',
                 r'$\kappa^P>\kappa^F$ and $\kappa^F$, $\kappa^P>0$',
                 r'$\kappa^P=\kappa^F$ and $\kappa^F$, $\kappa^P>0$',
                 r'$\kappa^P<\kappa^F$ and $\kappa^F$, $\kappa^P>0$',
                 r'$\kappa^F>0$ and $\kappa^P=0$',
                 ]
        le = ax.legend(custom_lines, names, title='Behaviour',
                     loc='upper left', bbox_to_anchor=(.01, 1),
                     ncol=1, prop = fontP)
        plt.setp(le.get_title(),fontsize=15)

        custom_lines = []
        for kpi in range(len(kpa)):
            custom_lines.append(Line2D([0], [0], color='k',
                                lw=0, marker=symb[kpi], markeredgecolor='k',
                                markersize=8))
        ax2 = ax.twinx()
        ax2.get_yaxis().set_visible(False)
        names = ['FS0','FS1','FS2','FS3']
        le = ax2.legend(custom_lines, names, title='strategy',
                     loc='upper left', bbox_to_anchor=(.35, 1),
                     ncol=1, prop = fontP)
        plt.setp(le.get_title(),fontsize=15)

        custom_lines = []
        for kpi in range(2):
            custom_lines.append(Line2D([0], [0], color='k',
                                lw=0, marker=symb[0], markeredgecolor='k',
                                markersize=8, markerfacecolor = 'w',
                                markeredgewidth=1+(1-kpi)*2))
        ax3 = ax2.twinx()
        ax3.get_yaxis().set_visible(False)
        names = ['0','0.01']
        le = ax3.legend(custom_lines, names, title='$\kappa^T$',
                     loc='upper left', bbox_to_anchor=(.5, 1),
                     ncol=1, prop = fontP)
        plt.setp(le.get_title(),fontsize=15)
    
fig, ax = plt.subplots(2, 1, figsize=(10,12))


size = 150

#legends
custom_lines = [Line2D([0], [0], color='k',
                lw=0, marker='P', markeredgecolor='k',
                markersize=12),
                Line2D([0], [0], color='tab:red',
                lw=0, marker='P', markeredgecolor='k',
                markersize=12),
                Line2D([0], [0], color='tab:blue',
                lw=0, marker='P', markeredgecolor='k',
                markersize=12)]
ax3 = ax[1].twinx()
ax3.get_yaxis().set_visible(False)
le = ax3.legend(custom_lines, ['BET', 'SKJ', 'YFT'], title='observation',
             loc='upper left', bbox_to_anchor=(0.01, 1), prop = fontP)
plt.setp(le.get_title(),fontsize=15)



ax[0].set_ylabel('Stomach fullness\n(dFAD associated; %)',
                 fontsize=fs-5)
ax[1].set_ylabel('Stomach fullness\n(dFAD associated; %)',
                 fontsize=fs-5)
ax[1].set_xlabel('Stomach fullness\n(not dFAD associated; %)',
                 fontsize=fs-5)

nfi = 5
nfi = 3

make_subplot(ax[1], title='(b) Bickley Jet flow', nfi=nfi, con='BJ', leg=False)

make_subplot(ax[0], title='(a) Double Eddy flow', nfi=nfi, con='DG')

plt.subplots_adjust(wspace = .05)
plt.savefig('figure3_nfi%d.png'%(nfi),bbox_inches='tight')
       
plt.show()
      
        
