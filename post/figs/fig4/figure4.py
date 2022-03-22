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
import pandas as pd
from numba import jit
import matplotlib.cm as mcm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, SymLogNorm

cmap = mcm.get_cmap('inferno')

fs = 18

thres = 18
thres0 = 0
n_bins = 10
x_bins = np.arange(n_bins)

fa = [0.,0.5, 1.,1.5]#[0,0.5, 1,1.5] # 0 0.5 1 1.5 # kappaF
kpa = [0.,0.5, 1.,1.5] #[0,0.5, 1,1.5] # 0 0.5 1 1.5 # kappaP
i = 0.01 # kappaI
pa = 0.1 # mean prey per grid cell at time=0

addx = np.array([-0.30, -0.1, 0.1, 0.3])
symb = ['o','^','s', 'p']

@jit(nopython=True)
def rrtimes(F, timestep = 0.5, rr=np.array([])):
    # time step 0.5 days
    # rr residence time steps at single FAD
    for j in range(F.shape[0]):
        ts = F[j]
        ts =  ts[ts>-10]
        if(len(ts)>0):
            if(ts[0]==0):
                ri = 0
            elif(ts[0]!=0):
                ri = 1
            for i in range(1,len(ts)):
                if(i==len(ts)-1 and ts[i-1]!=0):
                    if(ts[i]!=0 and ts[i]==ts[i-1]):
                        ri +=1
                        rr = np.append(rr,ri*timestep)
                    elif(ts[i]!=0 and ts[i]!=ts[i-1]):
                        rr = np.append(rr,ri*timestep)
                        rr = np.append(rr,1*timestep)
                    elif(ts[i]==0 and ts[i-1]!=0):
                        rr = np.append(rr,ri*timestep)
                elif(ts[i]!=0 and ts[i-1]==0):
                    ri = 1
                elif(ts[i]==ts[i-1] and ts[i]!=0):
                    ri += 1
                elif(ts[i]!=ts[i-1] and ts[i-1]!=0):
                    rr = np.append(rr,ri*timestep)
                    if(ts[i]!=0):
                        ri = 1
                    elif(ts[i]==0):
                        ri = 0
    return rr

def make_subplot(ax, dt=0.5, t=0., p=0., ylab='', title='', size=10,
                 con='DG'):
    ressCRT = np.full((len(fa),len(kpa), 20), np.nan)
    ressF = np.full((len(fa),len(kpa), 20), np.nan)
    ressFc = np.full((len(fa),len(kpa), 20), np.nan)

    for fi, f in enumerate(fa):
        for kpi, kp in enumerate(kpa):
            dirr = '/Users/nooteboom/Documents/tuna_project/dynamics/qd/'
            nc = Dataset(dirr+'output/FADrt%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa))
            nfc = 0
            ntc = 0
            
            nf, nt = (nc['FADs'][nfc:], nc['tuna'][ntc:])
            addxi = addx[kpi]
            for it in range(1):
                for nti in range(len(nt)):
                    for nfi in range(len(nf)):
                        nct = nc['Fa'][nfc+nfi,ntc+nti,it,:nf[nfi]].filled(-20)
                        rr =  rrtimes(nct)
                        ressF[fi, kpi, nfi] = nfi
                        ressFc[fi, kpi, nfi] = nf[nfi]
                        if(len(rr)>0):
                            ressCRT[fi, kpi, nfi] = np.nanmedian(rr)

    resE = []
    resF = []
    resP = []
    colors = []

    ax.set_yscale('log')
    if(con=='RW'):
        ax.set_ylabel(ylab, fontsize=fs)
    ax.set_title(title, fontsize=fs)

    for fi in range(len(fa)):
        for kpi in range(len(kpa)):
            if(kpa[kpi]==fa[fi] and fa[fi]>0):
                resE.append(ressCRT[fi,kpi])
            elif(kpa[kpi]>fa[fi]):
                resP.append(ressCRT[fi,kpi])
            elif(kpa[kpi]<fa[fi]):
                resF.append(ressCRT[fi,kpi])

    for ci in range(3):           
        colors.append(cmap(0.2+ci/3))
        
    ax.plot(ressF[0,0], np.nanmean(resP, axis=0), markersize = size, marker='o',
                      markeredgecolor='k', color=colors[0],
                      path_effects=[pe.Stroke(linewidth=5, foreground='k'),
                                    pe.Normal()])
    ax.plot(ressF[0,1] , np.nanmean(resE, axis=0), markersize = size, marker='o',
                      markeredgecolor='k', color=colors[1],
                      path_effects=[pe.Stroke(linewidth=5, foreground='k'),
                                    pe.Normal()])
    ax.plot(ressF[0,2] , np.nanmean(resF, axis=0), markersize = size, marker='o',
                      markeredgecolor='k', color=colors[2],
                      path_effects=[pe.Stroke(linewidth=5, foreground='k'),
                                    pe.Normal()])

    xt_minor = np.arange(len(nf)+1) - 0.5
    for x0, x1 in zip(xt_minor[::2], xt_minor[1::2]):
        ax.axvspan(x0, x1, color='black', alpha=0.1, zorder=0)
    xla = ['%.1f'%(i) for i in nf/0.98]
    ax.set_xticks(np.arange(len(nf)))
    ax.set_xticklabels(xla, fontsize=fs)
    ax.set_xlim(-0.5, len(nf)-0.5)
    ax.grid(False, axis="x")


def add_dat(res, f, kp, p, t, con, pa, pl, pi, it=0):
    dirr = '/Users/nooteboom/Documents/tuna_project/dynamics/qd/'
    nc = Dataset(dirr+'output/FADrt%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa))    
    nf, nt = (nc['FADs'][:], nc['tuna'][:])
    if(f==0.5 and kp==0.5 and t==0):
        print('number of FADs:',nf[nfi])
    nct = nc['Fa'][nfi,0,0,:nf[nfi]].filled(-20)
    rr =  rrtimes(nct)
    if(len(rr)>0 and it==1):
        res['CRT (days)'] = np.append(res['CRT (days)'],np.nanmedian(rr))
        res['strategy'] = np.append(res['strategy'], pl[pi])
        res['$\kappa^T$'] = np.append(res['$\kappa^T$'],t)
    else:
        idx = np.where(np.logical_and(res['strategy']== pl[pi],
                                      res['$\kappa^T$']==t))
        res['CRT (days)'][idx] += np.nanmedian(rr)
    return res

def sub2(ax, cax, res, ylabel='', cm = plt.get_cmap('cividis'), title='',
         con='RW'):
    res = pd.DataFrame.from_dict(res)
    res = res.pivot('$\kappa^T$','strategy','CRT (days)')
    b = sns.heatmap(res, ax=ax, annot=True, fmt=".1f",
                cbar_ax=cax,annot_kws={'fontsize':fs},
                vmin=0.5, vmax=25,
                #cmap=cm,
                norm=LogNorm(vmin=0.5, vmax=25, clip=True)
                )
    cax.set_ylabel('CRT (days)', fontsize=fs)
    b.set_xlabel('fishing strategy', fontsize=fs)
    b.set_ylabel('$\kappa^T$', fontsize=fs)
    if(con!='RW'):
        b.set_yticklabels(['',''], fontsize=fs)
        b.set_ylabel('', fontsize=fs)
    else:
        b.set_ylabel(ylabel+'\n\n'+'$\kappa^T$', fontsize=fs)
        plt.setp(ax.get_yticklabels(),fontsize=fs-3)
    return b

def make_subplot2(ax1, ax2, ax3, cax, nfi=4, kf=1,con='DG', title=''):
    par = [-2,-1,0,0.95]
    pl = ['FS0','FS1','FS2','FS3']
    ta = [0,0.01]
    
    res = {'CRT (days)':np.array([]),
           'strategy':np.array([]),
           '$\kappa^T$':np.array([])}
    resF = {'CRT (days)':np.array([]),
           'strategy':np.array([]),
           '$\kappa^T$':np.array([])}
    resP = {'CRT (days)':np.array([]),
           'strategy':np.array([]),
           '$\kappa^T$':np.array([])}
    resE = {'CRT (days)':np.array([]),
           'strategy':np.array([]),
           '$\kappa^T$':np.array([])}
    
    totF = 0
    totP = 0
    totE = 0
    for fi, f in enumerate(fa):
        for kpi, kp in enumerate(kpa):
            if(f>kp):
                totF += 1
            if(f<kp):
                totP += 1
            if(f==kp):
                totE += 1
            for pi, p in enumerate(par):
                for t in ta:
                    if(f>kp):
                        resF = add_dat(resF,f, kp, p, t, con, pa, pl, pi,
                                       it=totF)
                    elif(f<kp):
                        resP = add_dat(resP,f, kp, p, t, con, pa, pl, pi,
                                       it=totP)
                    else:
                        resE = add_dat(resE,f, kp, p, t, con, pa, pl, pi,
                                       it=totE)
    resP['CRT (days)'] /= totP
    resF['CRT (days)'] /= totF
    resE['CRT (days)'] /= totE
    sub2(ax1, cax, resF, ylabel='$\kappa^F$>$\kappa^P$',title=title,
             con=con)
    sub2(ax2, cax, resE, ylabel='$\kappa^F$=$\kappa^P$',title=title,
             con=con)
    b =sub2(ax3, cax, resP, ylabel='$\kappa^F$<$\kappa^P$',title=title,
             con=con)
    
    b.set_xlabel('fishing strategy', fontsize=fs)
    b.set_xticklabels(['FS0','FS1','FS2','FS3'], fontsize=fs)
                
 
def set_titles(ax, fs=15):
    ax[0,0].set_title('(d) Random Walk flow', fontsize=fs)
#    ax[1,0].set_title('(g)', fontsize=fs)
#    ax[2,0].set_title('(j)', fontsize=fs)
    ax[0,1].set_title('(e) Double Eddy flow', fontsize=fs)
#    ax[1,1].set_title('(h)', fontsize=fs)
#    ax[2,1].set_title('(k)', fontsize=fs)
    ax[0,2].set_title('(f) Bickley Jet flow', fontsize=fs)
#    ax[1,2].set_title('(i)', fontsize=fs)
#    ax[2,2].set_title('(l)', fontsize=fs) 
#%%
kt = 0.0
p = 0.95

nfi = 4

#grid_kws = {"width_ratios": (.9, 0.9, .03), "hspace": .5}
#fig, ax = plt.subplots(2,3, figsize=(16,8), gridspec_kw=grid_kws)

fig = plt.figure(figsize=(20,10))

outer_grid = fig.add_gridspec(2, 1, hspace=0.4,
                              height_ratios=[1,2])
inner_grid1 = outer_grid[0, 0].subgridspec(1, 4,
                                           width_ratios=[0.9,0.9,0.9, .03])
inner_grid2g = outer_grid[1, 0].subgridspec(1, 2,
                                           width_ratios=[4, .03],
                                           hspace=0.3,
                                           wspace=0.05)
inner_grid2 = inner_grid2g[0, 0].subgridspec(3, 3,
                                           #width_ratios=[.9, 0.9,0.9, .03],
                                           #hspace=0.3)
                                           hspace=0.05)
axs1 = inner_grid1.subplots()
axs2g = inner_grid2g.subplots()
axs2 = inner_grid2.subplots()
axs1[3].axis('off')
axs2g[0].axis('off')

#legends
custom_lines = []
for fi in range(3):
    custom_lines.append(Line2D([0], [0], color=cmap(0.2+fi/3),
                               lw=2, marker='o', markeredgecolor='k',
                               path_effects=[pe.Stroke(linewidth=5, foreground='k'),
                                    pe.Normal()]))
le1 = axs1[2].legend(custom_lines, [r'$\kappa^P>\kappa^F$',
                                    r'$\kappa^P=\kappa^F$',
                                    r'$\kappa^P<\kappa^F$'],
                     #title='$\kappa^F$',
             loc='upper left', bbox_to_anchor=(1.02, 0.8),
             fontsize=fs-3)

frame = le1.get_frame()
frame.set_facecolor('lightgray')

plt.setp(le1.get_title(),fontsize=fs)
plt.setp(le1.get_title(),fontsize=fs)
plt.setp(axs1[0].get_yticklabels(),fontsize=fs)
plt.setp(axs1[1].get_yticklabels(),fontsize=fs)
plt.setp(axs1[2].get_yticklabels(),fontsize=fs)

make_subplot(axs1[0], p=p, ylab='CRT (days)',
              title='(a) Random Walk flow', con='RW',
              t=kt)
make_subplot2(axs2[0,0],axs2[1,0],axs2[2,0], axs2g[1], con='RW',
              nfi = nfi,
              title='(d) Random Walk flow')

make_subplot(axs1[1], p=p, ylab='CRT (days)',
              title='(b) Double Eddy flow', con='DG',
              t=kt)
make_subplot2(axs2[0,1],axs2[1,1],axs2[2,1], axs2g[1], con='DG',
              nfi = nfi,
              title='(e) Double Eddy flow')

make_subplot(axs1[2], p=p, ylab='CRT (days)',
              title='(c) Bickley Jet flow', con='BJ',
              t=kt)
make_subplot2(axs2[0,2], axs2[1,2], axs2[2,2], axs2g[1], con='BJ',
              nfi = nfi,
              title='(f) Bickley Jet flow')

axs1[0].set_xlabel('dFAD density (10$^{-4}$ km$^{-2}$)', fontsize=fs) 
axs1[1].set_xlabel('dFAD density (10$^{-4}$ km$^{-2}$)', fontsize=fs) 
axs1[2].set_xlabel('dFAD density (10$^{-4}$ km$^{-2}$)', fontsize=fs) 
axs2[0,1].set_ylabel('')

def set_spines(ax):
    ax.grid(False, axis="x")
    [x.set_linewidth(1) for x in ax.spines.values()]
    [x.set_color('k') for x in ax.spines.values()]  
    [x.set_facecolor('k') for x in ax.spines.values()]  
    ax.set_yticks([1,10])
    ax.tick_params(which='major', axis='y',length=5, width=3)
    ax.tick_params(which='minor', axis='y',length=3, width=1)

set_spines(axs1[0])
set_spines(axs1[1])
set_titles(axs2)

#fig.patch.set_visible(True)
plt.savefig('figure4.pdf',bbox_inches='tight')
       
plt.show()
      
        
