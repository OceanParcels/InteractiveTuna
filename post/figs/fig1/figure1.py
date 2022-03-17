#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:42:18 2021

@author: nooteboom
"""
import numpy as np
import matplotlib.pylab as plt
from copy import copy
import cmocean
from netCDF4 import Dataset
from scipy.stats import multivariate_normal
from scipy.stats import norm
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
import matplotlib
import math
sns.set()

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def LogisticCurve(x, L=1, k=0.5, x0=8): # x is the number of neighbouring tuna
    res = 1 + L / (1+math.e**(-k*(x-x0)))
    return res

def Logit(x, L=1, x0 = 8, k0=0.5):
    return -1*np.log(L/(x-1))/k0+x0

def ass_num(lonF, latF, lon, lat, R=2):
    # return the number of associated tuna per FAD
    nas = np.zeros(lonF.shape)
    for n in range(len(nas)):
        nas[n] =np.sum(((lonF[n]-lon)**2+(latF[n]-lat)**2)**0.5 < R) 
    return nas

# domain size
Lx = 140
Ly = 70
res = 10 # domain resolutiom

#%% DG
daysec = 24*3600
eps = 0.2
# frequency
om = 1/(daysec) * 2 * np.pi
A= 0.1

def f(x,t, eps=eps, om=om):
    f1 = eps*np.sin(om*t)*x[0]**2
    f2 = (1-2*eps*np.sin(om*t))*x[0]
    return f1 + f2

def dfdx(x,t, eps=eps, om=om):
    f1 = 2*eps*np.sin(om*t)*x[0]
    f2 = (1-2*eps*np.sin(om*t))
    return f1 + f2

def scaleX_DG(x, ly=Ly):
    x0 = x[0] / ly
    x1 = x[1] / ly
    return [x0, x1]

def vDG(x,t, A=A):
    res = [0, 0]
    xc = [0, 0]
    xc[0], xc[1] = scaleX_DG(x)
    res[0] = -np.pi*A*np.sin(np.pi*f(xc,t))* np.cos(np.pi*xc[1])
    res[1] = np.pi*A*np.cos(np.pi*f(xc,t)) * np.sin(np.pi*xc[1]) * dfdx(xc,t)
    return res

def create_preyfieldDATADG(lx, ly, res, nprey=int(1e5)):
    # Randomly distribute the prey over the grid
    dataP = np.zeros(((ly//res)+2, (lx//res)+2))
    for n in range(nprey):
        i = 1+np.random.binomial(dataP.shape[0]-3, 0.5)
        j = 1+np.random.binomial(dataP.shape[1]-2, 0.3)
        dataP[i,j] += 1
    # normalize the field
    dataP /= dataP.max()
    dataP += 0.1
    dataP *= 0.5
    assert dataP.max() <= 1
    assert dataP.min() >= 0

    return dataP


def fstreamDG(y,t):
    x1, x2 = scaleX_DG(y)
    y = [x1,x2]
    return A*np.sin(np.pi*f(y,t))*np.sin(np.pi*y[1])

preyFDG = create_preyfieldDATADG(Lx, Ly, res)
#%% BJ
def create_preyfieldDATABJ(lx, ly, res, nprey=int(1e5)):
    # Randomly distribute the prey over the grid
    dataP = np.zeros(((ly//res)+2, (lx//res)+2))
    for n in range(nprey):
        i = 1+np.random.binomial(dataP.shape[0]-3, 0.5)
        j = np.random.randint(1,dataP.shape[1]-1)
        dataP[i,j] += 1
    # normalize the field
    dataP = dataP / dataP.max() * 0.4
    assert dataP.max() <= 1
    assert dataP.min() >= 0

    return dataP


r0 = 6371.

preyFBJ = create_preyfieldDATABJ(Lx, Ly, res)

#Parameters for the Bickley jet
Ubj = 0.06266 #1./1000 #600#0.06266 # km/s
L = 1770 # km
r0 = 6371 # km

k1 = 2 * 1/ r0
k2 = 2 * 2 / r0
k3 = 2 * 3/ r0
eps1 = 0.075
eps2 = 0.4
eps3 = 0.3
c3 = 0.461 * Ubj
c2 = 0.205 * Ubj
c1 = 0.1446 * Ubj

def scale_locBJ(y, Lx=Lx, Ly=Ly):
    x0 = y[0] / Lx * np.pi * r0
    y0 = (y[1] - Ly/2) / (Ly/2) * 3e3
    return [x0, y0]

def vBJ(y,t): #2D velocity Bickley Jet
    t *= 300
    x1, x2 = scale_locBJ(y)
    
    f1 = eps1 * np.exp(-1j *k1 * c1 * t)
    f2 = eps2 * np.exp(-1j *k2 * c2 * t)
    f3 = eps3 * np.exp(-1j *k3 * c3 * t)
    F1 = f1 * np.exp(1j * k1 * x1)
    F2 = f2 * np.exp(1j * k2 * x1)
    F3 = f3 * np.exp(1j * k3 * x1)    
    G = np.real(np.sum([F1,F2,F3]))
    G_x = np.real(np.sum([1j * k1 *F1, 1j * k2 * F2, 1j * k3 * F3]))    
    u =  Ubj / (np.cosh(x2/L)**2)  +  2 * Ubj * np.sinh(x2/L) / (np.cosh(x2/L)**3) *  G
    v = Ubj * L * (1./np.cosh(x2/L))**2 * G_x  
    return [u,v]

def fstreamBJ(y,t):
    x1, x2 = scale_locBJ(y)
    s0 = -Ubj*L*np.tanh(x2/L)
    f1 = eps1 * np.exp(-1j *k1 * c1 * t)
    f2 = eps2 * np.exp(-1j *k2 * c2 * t)
    f3 = eps3 * np.exp(-1j *k3 * c3 * t)
    F1 = f1 * np.exp(1j * k1 * x1)
    F2 = f2 * np.exp(1j * k2 * x1)
    F3 = f3 * np.exp(1j * k3 * x1)
    s1 = Ubj*L * 1/np.cosh(x2/L)**2* np.real(np.sum([F1,F2,F3]))
    return s0 + s1
    
#%%
fs = 18
Fcmap = 'cool_r'
fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True,
                       figsize=(15,10),
                               gridspec_kw={'width_ratios': [40, 40]})        

        
        
xs, ys = np.meshgrid(np.arange(0,Lx+2*res,res),np.arange(0,Ly+2*res,res))
xs = xs - res/2
ys = ys - res/2


xss, yss = np.meshgrid(np.arange(0,Lx+2*res,res/10),np.arange(0,Ly+2*res,res/10))
streamDG = np.zeros((2,xss.shape[0],xss.shape[1]))
streamBJ = np.zeros((2,xss.shape[0],xss.shape[1]))

loopt = np.arange(0,1e6,2e3)
props = dict(boxstyle='round', facecolor=None, alpha = 0)
bo = True
for ti in range(len(loopt)):
    time = loopt[ti]
    if('%.1f'%(time/24/3600)=='5.0' and bo):
        bo = False
        for i in range(xss.shape[0]):
            for j in range(xss.shape[1]):
                st = fstreamDG([xss[i,j],yss[i,j]], time)
                streamDG[0,i,j] = st
                st = fstreamBJ([xss[i,j],yss[i,j]], time)
                streamBJ[0,i,j] = st
                

        
        ax[0,0].set_xlim(0,Lx)
        ax[0,0].set_ylim(0,Ly)
        ax[0,0].text(1.02, 1.07, '%.1f days'%(time/24/3600),
                   transform=ax[0,0].transAxes, fontsize=fs,
                   verticalalignment='top', bbox=props)
        ax[0,0].set_title('(a) Double Gyre', fontsize=fs)
        im0 = ax[0,0].pcolormesh(xs, ys, preyFDG, vmin=0,vmax=1,
                         cmap=cmocean.cm.algae, shading='nearest')
        ax[0,0].contour(xss, yss, streamDG[0], 15, colors='k', linewidths=2)
        
        ax[0,1].set_xlim(0,Lx)
        ax[0,1].set_ylim(0,Ly)
        ax[0,1].set_title('(b) Bickley Jet', fontsize=fs)
        im0 = ax[0,1].pcolormesh(xs, ys, preyFBJ, vmin=0,vmax=1,
                         cmap=cmocean.cm.algae, shading='nearest')
        ax[0,1].contour(xss, yss, streamBJ[0], 
                      levels= [-150, -145,-140,-130,-120,-115, -110, -100, -90, -60, -30,
                               0, 30, 60, 90, 100, 110,115, 120,130,140,145,150], 
                      colors='k', linewidths=2)
   
cbar_ax = fig.add_axes([0.93, 0.55, 0.02, 0.32])
cbar = fig.colorbar(im0, cax=cbar_ax)
cbar.set_label('prey', fontsize=fs)     

#%% model snapshot
FADb = True

R = 2 # FAD association radius
Dt = 4*1.2e3 # the output timestep (s)
npart = 316 # total number of particles
nfad = 15 # number of FADs

# domain size
lx = 140
ly = 70
int_dist = 10

ms = 10
ms2 = 30
dirr = 'input/'
ncf = Dataset(dirr+'FADPreyDG_no0_npart%d_nfad%d_T0.01_F0.50_P0.50_I0.01_p0.9_Pa0.1.nc'%(npart,nfad))

obs = ncf['lon'][:].shape[1]
if(nfad>0 and FADb):
    idxFAD = (ncf['ptype'][:]==1)
    idxP = (ncf['ptype'][:]==2)
    idxO = (np.logical_or(ncf['ptype'][:]==1,ncf['ptype'][:]==2))
    print('number of FADs: ', np.sum(idxFAD))
    FADlons = ncf['lon'][(idxFAD, np.arange(obs))]
    FADlats = ncf['lat'][(idxFAD, np.arange(obs))]
    
    FADkaps = ncf['FADkap'][(idxFAD, np.arange(obs))]
    no_as = np.around(Logit(FADkaps)) # number of associated tuna at FAD
    
    if(np.sum(idxP)>0):
        preylons = ncf['lon'][(idxP, np.arange(obs))]
        preylats = ncf['lat'][(idxP, np.arange(obs))]
    
    Olons = ncf['lon'][(idxO, np.arange(obs))]
    Olats = ncf['lat'][(idxO, np.arange(obs))]
    lons = ncf['lon'][~idxO]
    lats = ncf['lat'][~idxO]
else:
    lons = ncf['lon'][:]
    lats = ncf['lat'][:]

ptype = ncf['ptype'][:]

patches1 = [Line2D([0], [0], marker='o', color='lightgray', label='Scatter',
                  markerfacecolor=matplotlib.cm.get_cmap(Fcmap)(0),
                  markersize=ms, markeredgecolor='k', linewidth=0),
           Line2D([0], [0], marker='o', color='lightgray', label='Scatter',
                  markerfacecolor='gold', markersize=ms, markeredgecolor='k',
                  linewidth=0)
           ]
texts1=['FAD', 'Tuna']
patches2 = [mpatches.Circle((0.5, 0.5), 1,
                           edgecolor='k',
                           linestyle='--',
                           facecolor='lightgray', 
                           linewidth=2),
            mpatches.Circle((0.5, 0.5), 1,
                           edgecolor='k',
                           facecolor='lightgray', 
                           linewidth=2)
           ]
texts2=['FAD attraction radius','FAD association radius']

for its in range(1,lons.shape[1]):
    if(its==50): 
        ncP = Dataset(dirr+'FADPreyDG_no0_npart%d_nfad%d_T0.01_F0.50_P0.50_I0.01_p0.9_Pa0.1_%.4dprey.nc'%(npart,
                                                                            nfad,
                                                                            its))
        prey = ncP['prey'][0,0,1:-1,1:-1]
        preylon = ncP['nav_lon'][1:-1,1:-1]
        preylat = ncP['nav_lat'][1:-1,1:-1]

        im0 = ax[1,0].pcolormesh(preylon, preylat, prey, vmin=0,vmax=1,
                             cmap=cmocean.cm.algae, shading='nearest')
        if(nfad>0 and FADb):
    
            for i in range(FADlons.shape[0]):
                ax[1,0].add_artist(plt.Circle((FADlons[i,its], FADlats[i,its]),
                                         int_dist,
                                         fill = False,
                                         edgecolor='k',
                                         linestyle='--',
                                         lw = 2,
                                         zorder = 10000,
                                         label='FAD attraction radius'))
    
        ax[1,0].scatter([lons[:,its]], [lats[:,its]], c='gold',edgecolor='k',
                   s=ms2, label='tuna')
        if(nfad>0 and FADb):
            ass_tun = ass_num(FADlons[:,its], FADlats[:,its],
                              lons[:,its], lats[:,its], 
                              R)
            im = ax[1,0].scatter([FADlons[:,its]], [FADlats[:,its]],
                       c=ass_tun,
                       cmap=Fcmap, edgecolor='k',
                       vmin=1, vmax=10,
                       s=ms2, label='FAD', zorder=10001)
        
            if(R>0):
                for i in range(FADlons.shape[0]):
                    ax[1,0].add_artist(plt.Circle((FADlons[i,its], FADlats[i,its]),
                                             R ,
                                             fill = False,
                                             edgecolor='k',
                                             zorder = 10000))
            
        ax[1,0].legend(patches1, texts1, fontsize=fs, edgecolor='k',
                     bbox_to_anchor=(0.2, -0.15),
                     loc='upper left', ncol=2,
                     handler_map={mpatches.Circle: HandlerEllipse()}).get_frame().set_facecolor('lightgray')
    
        ax[1,1].legend(patches2, texts2, fontsize=fs,
                   edgecolor='k', bbox_to_anchor=(0.2, -0.15),
                     loc='upper left', ncol=1,
                     handler_map={mpatches.Circle: HandlerEllipse()}).get_frame().set_facecolor('lightgray')

        if(nfad>0):
            cbaxes = fig.add_axes([0.93, 0.15, 0.02, 0.32])
            cbar2 = fig.colorbar(im, extend='max', cax = cbaxes)
            cbar2.set_label('# tuna associated with FAD', fontsize=fs)

#%% And the bickley jet
npart = 306 # total number of particles
nfad = 5 # number of FADs
dirr = 'input/'
ncf = Dataset(dirr+'FADPreyBJ_no0_npart%d_nfad%d_T0.01_F0.50_P0.50_I0.01_p0.1_Pa0.2.nc'%(npart,nfad))

obs = ncf['lon'][:].shape[1]
if(nfad>0 and FADb):
    idxFAD = (ncf['ptype'][:]==1)
    idxP = (ncf['ptype'][:]==2)
    idxO = (np.logical_or(ncf['ptype'][:]==1,ncf['ptype'][:]==2))
    print('number of FADs: ', np.sum(idxFAD))
    FADlons = ncf['lon'][(idxFAD, np.arange(obs))]
    FADlats = ncf['lat'][(idxFAD, np.arange(obs))]
    
    FADlons = np.concatenate((FADlons-lx,FADlons,FADlons+lx), axis=0)
    FADlats = np.concatenate((FADlats,FADlats,FADlats), axis=0)

    FADkaps = ncf['FADkap'][(idxFAD, np.arange(obs))]
    no_as = np.around(Logit(FADkaps)) # number of associated tuna at FAD
    
    if(np.sum(idxP)>0):
        preylons = ncf['lon'][(idxP, np.arange(obs))]
        preylats = ncf['lat'][(idxP, np.arange(obs))]
    
    Olons = ncf['lon'][(idxO, np.arange(obs))]
    Olats = ncf['lat'][(idxO, np.arange(obs))]
    
    lons = ncf['lon'][~idxO]
    lats = ncf['lat'][~idxO]
else:
    lons = ncf['lon'][:]
    lats = ncf['lat'][:]

ptype = ncf['ptype'][:]

for its in range(1,lons.shape[1]):
    if(its==90):
        ncP = Dataset(dirr+'FADPreyBJ_no0_npart%d_nfad%d_T0.01_F0.50_P0.50_I0.01_p0.1_Pa0.2_%.4dprey.nc'%(npart,
                                                                            nfad,
                                                                            its))
        prey = ncP['prey'][0,0,1:-1,1:-1]
        preylon = ncP['nav_lon'][1:-1,1:-1]
        preylat = ncP['nav_lat'][1:-1,1:-1]

        im0 = ax[1,1].pcolormesh(preylon, preylat, prey, vmin=0,vmax=1,
                             cmap=cmocean.cm.algae, shading='nearest')
        if(nfad>0 and FADb):
    
            for i in range(FADlons.shape[0]):
                ax[1,1].add_artist(plt.Circle((FADlons[i,its], FADlats[i,its]),
                                         int_dist,
                                         fill = False,
                                         edgecolor='k',
                                         linestyle='--',
                                         lw = 2,
                                         zorder = 10000,
                                         label='FAD attraction radius'))
    
        ax[1,1].scatter([lons[:,its]], [lats[:,its]], c='gold',edgecolor='k',
                   s=ms2, label='tuna')
        if(nfad>0 and FADb):
            ass_tun = ass_num(FADlons[:,its], FADlats[:,its],
                              lons[:,its], lats[:,its], 
                              R)
            im = ax[1,1].scatter([FADlons[:,its]], [FADlats[:,its]],
                       c=ass_tun,
                       cmap=Fcmap, edgecolor='k',
                       vmin=1, vmax=10,
                       s=ms2, label='FAD', zorder=10001)
        
            if(R>0):
                for i in range(FADlons.shape[0]):
                    ax[1,1].add_artist(plt.Circle((FADlons[i,its], FADlats[i,its]),
                                             R ,
                                             fill = False,
                                             edgecolor='k',
                                             zorder = 10000))

#%% general

ax[1,0].set_xlabel('x (km)', fontsize=fs)
ax[1,1].set_xlabel('x (km)', fontsize=fs)
ax[1,0].set_ylabel('y (km)', fontsize=fs)
ax[0,0].set_ylabel('y (km)', fontsize=fs)

ax[1,0].set_title('(c)', fontsize=fs)
ax[1,1].set_title('(d)', fontsize=fs)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.savefig('figure1.png', bbox_inches='tight')
plt.show()