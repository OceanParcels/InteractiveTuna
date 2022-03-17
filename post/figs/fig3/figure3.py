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

cmap = mcm.get_cmap('inferno')

fa = [0.,0.5, 1.,1.5]# kappaF
kpa = [0.,0.5, 1.,1.5]# kappaP

fs = 20

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

    ax.scatter(dic['BETfm'],dic['BETFm'],marker = 'P', c=cos[0], s=160, edgecolor='k')
    ax.scatter(dic['SKJfm'],dic['SKJFm'],marker = 'P', c=cos[1], s=160, edgecolor='k')
    ax.scatter(dic['YFTfm'],dic['YFTFm'],marker = 'P', c=cos[2], s=160, edgecolor='k')

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

def make_subplot(ax, con='BJ', t=0.01, i=0.01, p=0, nfi = 5, pa=0.1,
                 title='', leg=True):
    symb = ['o','^','s', 'p']
    ax.plot([0,100],[0,100], '--', linewidth=2, c='k', alpha=0.8)
    dirr = 'input/'
    nc = Dataset(dirr+'FADst%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,0,0,i,p,pa))
    nf, nt = (nc['FADs'][:], nc['tuna'][:])
    
    print('number of FADs:',nf[nfi])
    
    Sta = np.zeros((len(fa), len(kpa)))
    Stna = np.zeros((len(fa), len(kpa)))
    for fi, f in enumerate(fa):
        for kpi, kp in enumerate(kpa):
            nc = Dataset(dirr + 'FADst%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa))
            nf, nt = (nc['FADs'][:], nc['tuna'][:])#np.meshgrid
            for it in range(1):
                for nti in range(len(nt)):
                    Sta[fi,kpi], Stna[fi,kpi] = fStomach2(nc, nf, nfi,
                                                                      nt, nti) 
    # from index to percentage
    Sta *= 100
    Stna *= 100   
    
    for fi in range(len(fa)):
        color = cmap(0.2+fi/len(fa))
        if(fa[fi]>0):
            ax.scatter(Stna[fi,0], Sta[fi,0], s = size, marker=symb[0],
                       edgecolor='k', color=color, alpha=0.7)
        ax.scatter(Stna[fi,1] , Sta[fi,1], s = size, marker=symb[1],
                     edgecolor='k', color=color, alpha=0.7)
        ax.scatter(Stna[fi,2], Sta[fi,2], s = size, marker=symb[2],
                      edgecolor='k', color=color, alpha=0.7)
        ax.scatter(Stna[fi,3], Sta[fi,3], s = size, marker=symb[3],
                      edgecolor='k', color=color, alpha=0.7)

    ax.set_title(title, fontsize=fs)
    plot_obs(ax)
    
    if(leg):
        #legend
        custom_lines = []
        for fi in range(len(fa)):
            custom_lines.append(Line2D([0], [0], color=cmap(0.2+fi/len(fa)),
                                lw=0, marker='o', markeredgecolor='k',
                                markersize=8, alpha=0.7))
        ax.legend(custom_lines, fa, title='$\kappa^F$',
                     loc='upper left', bbox_to_anchor=(.01, 1),
                     ncol=1)
        custom_lines = []
        for kpi in range(len(kpa)):
            custom_lines.append(Line2D([0], [0], color='k',
                                lw=0, marker=symb[kpi], markeredgecolor='k',
                                markersize=8))
        ax2 = ax.twinx()
        ax2.get_yaxis().set_visible(False)
        ax2.legend(custom_lines, fa, title='$\kappa^P$',
                     loc='upper left', bbox_to_anchor=(.17, 1),
                     ncol=1)

def make_subplot2(ax, con='BJ', Pa=0.1, nfi=2, title='',
                 kp = 0.5, kf = 1.5, i=0.01,
                 cmap=mcm.get_cmap('Spectral'), leg=True):
    symb = ['D','X','s', 'p']
    
    dirr = 'input/'
    nc = Dataset(dirr+'FADst%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,0.01,0,0,i,0.95,Pa))
    nf, nt = (nc['FADs'][:], nc['tuna'][:])
    print('number of FADs: ',nf[nfi])
    ta = [0, 0.01]
    pa = [-2,-1,0,0.95]
    Sta = np.zeros((len(pa), len(ta)))
    Stna = np.zeros((len(pa), len(ta)))
    for pi, p in enumerate(pa):
        for ti, t in enumerate(ta):
            nc = Dataset(dirr + 'FADst%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,kf,kp,i,p,Pa))
            nf, nt = (nc['FADs'][:], nc['tuna'][:])
            for it in range(1):
                for nti in range(len(nt)):
                    Sta[pi,ti], Stna[pi,ti] = fStomach2(nc, nf, nfi, nt, nti) 
    # from index to percentage
    Sta *= 100
    Stna *= 100   
    
    for pi in range(len(pa)):
        color = cmap(0.2+pi/len(pa))
        ax.scatter(Stna[pi,0], Sta[pi,0], s = size, marker=symb[0],
                      edgecolor='k', color=color, alpha=0.7)
    for pi in range(len(pa)):
        print(pa[pi])
        print(Stna[pi,1] , Sta[pi,1])
        color = cmap(0.2+pi/len(pa))
        ax.scatter(Stna[pi,1] , Sta[pi,1], s = size, marker=symb[1],
                     edgecolor='k', color=color, alpha=0.7)
        
        
    ax.plot([0,100],[0,100], '--', linewidth=2, c='k', alpha=0.8, zorder=0)
    ax.set_title(title, fontsize=fs)
    plot_obs(ax)

    if(leg):
        #legends
        custom_lines = []
        for fi in range(len(pa)):
            custom_lines.append(Line2D([0], [0], color=cmap(0.2+fi/len(pa)),
                                lw=0, marker='o', markeredgecolor='k',
                                markersize=8))
        ax.legend(custom_lines, ['FS0','FS1','FS2','FS3'],# title='p',
                     loc='upper left', bbox_to_anchor=(.01, 1),
                     ncol=1)
        custom_lines = []
        for kpi in range(len(ta)):
            custom_lines.append(Line2D([0], [0], color='k',
                                lw=0, marker=symb[kpi], markeredgecolor='k',
                                markersize=8))
        ax2 = ax.twinx()
        ax2.get_yaxis().set_visible(False)
        ax2.legend(custom_lines, [0, 0.01], title='$\kappa^T$',
                     loc='upper left', bbox_to_anchor=(.18, 1),
                     ncol=1)
    ax.set_ylim(0,70)
    ax.set_xlim(0,70)
    
fig, ax = plt.subplots(2, 2, figsize=(14,8), sharey = True, sharex=True)      


size = 150

#legends
if(True):
    custom_lines = [Line2D([0], [0], color='k',
                    lw=0, marker='P', markeredgecolor='k',
                    markersize=12),
                    Line2D([0], [0], color='tab:red',
                    lw=0, marker='P', markeredgecolor='k',
                    markersize=12),
                    Line2D([0], [0], color='tab:blue',
                    lw=0, marker='P', markeredgecolor='k',
                    markersize=12)]
    ax3 = ax[1,0].twinx()
    ax3.get_yaxis().set_visible(False)
    ax3.legend(custom_lines, ['BET', 'SKJ', 'YFT'], title='observation',
                 loc='upper right', bbox_to_anchor=(1.15, -.102))
else:
    custom_lines = [Line2D([0], [0], color='k',
                    lw=2,markersize=0),
                    Line2D([0], [0], color='tab:red',
                           linestyle='--',
                    lw=2,markersize=0),
                    Line2D([0], [0], color='tab:blue',
                    lw=2,markersize=0)]
    ax3 = ax[0].twinx()
    ax3.get_yaxis().set_visible(False)
    ax3.legend(custom_lines, ['BET','SKJ', 'YFT'], title='observations',
                 loc='upper right', bbox_to_anchor=(1.15, -.102))



ax[0,0].set_ylabel('Stomach fullness\n(dFAD associated; %)',
              fontsize=fs-5)
ax[1,0].set_ylabel('Stomach fullness\n(dFAD associated; %)',
              fontsize=fs-5)
ax[1,0].set_xlabel('Stomach fullness\n(not dFAD associated; %)', fontsize=fs-5)
ax[1,1].set_xlabel('Stomach fullness\n(not dFAD associated; %)', fontsize=fs-5)

nfi = 3
nfi = 5

t = 0.01
p = -1

kp = 1
kf = 1

make_subplot(ax[1,0], title='(c)', nfi=nfi, t=t, p=p, con='BJ', leg=False)
make_subplot2(ax[1,1], title='(d)', nfi=nfi, kp=kp, kf=kf, con='BJ', leg=False)
plt.figtext(0.5,0.95, "Double Eddy flow", ha="center", va="top", fontsize=fs)
plt.figtext(0.5,0.5, "Bickley Jet flow", ha="center", va="top", fontsize=fs)

make_subplot(ax[0,0], title='(a)', nfi=nfi, t=t, p=p, con='DG')
make_subplot2(ax[0,1], title='(b)', nfi=nfi, kp=kp, kf=kf, con='DG')

plt.subplots_adjust(wspace = .05)
plt.savefig('figure3.png',bbox_inches='tight')
       
plt.show()
      
        
