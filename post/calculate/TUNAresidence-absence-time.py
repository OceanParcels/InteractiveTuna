import numpy as np
from netCDF4 import Dataset
import sys
from numba import jit

@jit(nopython=True)
def rest(lon, lat, nf, Ra=2):
    lont = lon[nf+1:]
    latt = lat[nf+1:]
    lonf = lon[1:nf+1]
    latf = lat[1:nf+1]
    res = np.zeros(lont.shape)
    for t in range(lont.shape[1]):
        for p in range(lont.shape[0]):
            for fp in range(lonf.shape[0]):
                dist = ((lont[p,t]-lonf[fp,t])**2+(latt[p,t]-latf[fp,t])**2)
                if(dist<Ra**2):
                    res[p,t] = fp + 1
    return res


def FADrt(ncf, nf, nt):
    lon = ncf['lon'][:]
    lat = ncf['lat'][:]
    res = rest(lon, lat, nf)
    return res

if(__name__=='__main__'):
    con = 'BJ'
    assert con in ['RW','DG','BJ']
    
    t = float(sys.argv[1])#0.01 # kappT
    f = float(sys.argv[2]) # 0 0.5 1 1.5 # kappaF
    kp = float(sys.argv[3]) # 0 0.5 1 1.5 # kappaP
    i = 0.01 # kappaI
    p = float(sys.argv[5]) # 0 0.95 -1 # fishing strategy
    pa = float(sys.argv[6]) # mean prey per grid cell at time=0
    
    its = 1 # Monte Carlo iterations
    # number of FAD particless
    nf = np.array([2, 5,10, 15,20, 30, 40])
    # number of tuna particles
    nt = np.array([500])
    
    dirR = '/nethome/3830241/tuna_project/quantify_cases/%s/'%(con)
    
    ds = Dataset('output/FADrt%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa), 'w', format='NETCDF4')
    
    vits = ds.createDimension('MCits', its)
    vnf = ds.createDimension('FADs', len(nf))
    vnt = ds.createDimension('tuna', len(nt))
    vnt2 = ds.createDimension('Tno', nt[-1])
    vtime = ds.createDimension('time', 201)
    
    snf = ds.createVariable('FADs', int, ('FADs',))
    stf = ds.createVariable('tuna', int, ('tuna',))
    stime = ds.createVariable('time', 'f4', ('time',))
    stime.units = 'days'
    
    sFrt =  ds.createVariable('Fa', int, ('FADs','tuna','MCits','Tno','time',))
    
    snf[:] = nf
    stf[:] = nt
    stime[:] = np.arange(201) / 2
    
    for nfi in range(len(nf)):
        for nti in range(len(nt)):
            for iti in range(its):
                npart = nf[nfi] + nt[nti] + 1
                ncr = Dataset(dirR+'output/FADPrey%s_no%d_npart%d_nfad%d_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,iti,npart,nf[nfi],t,f,kp,i,p,pa))
                sFrt[nfi,nti,iti,:nt[nti]] = FADrt(ncr, nf[nfi], nt[nti])
    ds.close()
