import numpy as np
from netCDF4 import Dataset
import sys

def FADsta(ncf, nf):
    # stomach fullness of FAD associated tuna
    sta = ncf['Sta'][:]
    stac = ncf['Stac'][:]
    res = np.full(sta.shape[0]-nf-1, np.nan)
    for i in range(sta.shape[0]-nf-1):
        if(sta[nf+i+1, -1]>0 and stac[nf+i+1, -1]>0):
            res[i] = sta[nf+i+1, -1] / stac[nf+i+1,-1]
    return res

def FADstna(ncf, nf):
    # stomach fullness of free school tuna
    sta = ncf['Stna'][:]
    stac = ncf['Stnac'][:]
    res = np.full(sta.shape[0]-nf-1, np.nan)
    for i in range(sta.shape[0]-nf-1):
        if(sta[nf+i+1, -1]>0 and stac[nf+i+1, -1]>0):
            res[i] = sta[nf+i+1, -1] / stac[nf+i+1, -1]
    return res

if(__name__=='__main__'):
    con = 'BJ'
    assert con in ['RW','DG','BJ']
    
    t = float(sys.argv[1]) # kappaT
    f = float(sys.argv[2]) # 0 0.5 1 1.5 # kappaF
    kp = float(sys.argv[3]) # 0 0.5 1 1.5 # kappaP
    i = 0.01 # kappaI
    p = float(sys.argv[5]) # 0 0.95 -1 # fishing strategy
    pa = float(sys.argv[6]) # mean prey per grid cell at time=0
    
    its = 1 # Monte Carlo iterations
    # number of FAD particless
    nf = np.array([0, 2, 5, 10, 15, 20, 25, 30, 35, 40])
    # number of tuna particles
    nt = np.array([500])
    
    dirR = '/nethome/3830241/tuna_project/quantify_cases/%s/'%(con)
    
    ds = Dataset('output/FADst%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa), 'w', format='NETCDF4')
    
    vits = ds.createDimension('MCits', its)
    vnf = ds.createDimension('FADs', len(nf))
    vnt = ds.createDimension('tuna', len(nt))
    vnt2 = ds.createDimension('Tno', nt[-1])
    vtime = ds.createDimension('time', 201)
    
    snf = ds.createVariable('FADs', int, ('FADs',))
    stf = ds.createVariable('tuna', int, ('tuna',))
    stime = ds.createVariable('time', 'f4', ('time',))
    stime.units = 'days'
    
    sFrt =  ds.createVariable('Tsta', float, ('FADs','tuna','MCits','Tno'))
    sFa =  ds.createVariable('Tstna', float, ('FADs','tuna','MCits','Tno'))
    
    snf[:] = nf
    stf[:] = nt
    stime[:] = np.arange(201) / 2
    
    for nfi in range(len(nf)):
        for nti in range(len(nt)):
            for iti in range(its):
                npart = nf[nfi] + nt[nti] + 1
                ncr = Dataset(dirR+'output/FADPrey%s_no%d_npart%d_nfad%d_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,iti,npart,nf[nfi],t,f,kp,i,p,pa))
                sFrt[nfi,nti,iti,:nt[nti]] = FADsta(ncr, nf[nfi])
                sFa[nfi,nti,iti,:nt[nti]] = FADstna(ncr, nf[nfi])
    ds.close()
