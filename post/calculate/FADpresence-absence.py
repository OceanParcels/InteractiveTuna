import numpy as np
from netCDF4 import Dataset
import sys
from numba import jit

@jit(nopython=True)
def calc_at(lonF, latF, lonT, latT, res, R=2):
    for i in range(lonF.shape[0]):
        for t in range(lonF.shape[1]):
            dist = ((lonF[i,t]-lonT[:,t])**2+(latF[i,t]-latT[:,t])**2)**(0.5)
            res[i,t] = np.sum(dist<=R)
    return res

def FADnt_temp(ncf, nf):
    # total number of tune caught near FAD
    resb =  ncf['caught'][:1+nf]

    lonF = ncf['lon'][1:1+nf]
    latF = ncf['lat'][1:1+nf]
    lonT = ncf['lon'][1+nf:]
    latT = ncf['lat'][1+nf:]
    resa = np.zeros(lonF.shape)

    resa = calc_at(lonF, latF, lonT, latT, resa)

    return resa, resb

def FADnt(ncf, nf):
    # number of associated tuna per FAD
    resa = ncf['FADkap'][1:1+nf]
    # total number of tune caught near FAD
    resb =  ncf['caught'][1:1+nf]

    return resa, resb

if(__name__=='__main__'):
    con = 'BJ'
    assert con in ['RW','DG','BJ']
    
    t = float(sys.argv[1]) # kappT
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
    
    # Write the output
    ds = Dataset('output/FADpa%s_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,t,f,kp,i,p,pa), 'w', format='NETCDF4')
    
    vits = ds.createDimension('MCits', its)
    vnf = ds.createDimension('FADs', len(nf))
    vnf2 = ds.createDimension('FADno', nf[-1]+1)
    vnt = ds.createDimension('tuna', len(nt))
    vtime = ds.createDimension('time', 201)
    
    snf = ds.createVariable('FADs', int, ('FADs',))
    stf = ds.createVariable('tuna', int, ('tuna',))
    stime = ds.createVariable('time', 'f4', ('time',))
    stime.units = 'days'
    
    sFp =  ds.createVariable('Fa', int, ('FADs','tuna','MCits','FADno','time',))
    sFc =  ds.createVariable('Fc', int, ('FADs','tuna','MCits','FADno','time',))
    
    snf[:] = nf
    stf[:] = nt
    stime[:] = np.arange(201) / 2
    
    for nfi in range(len(nf)):
        for nti in range(len(nt)):
            for iti in range(its):
                npart = nf[nfi] + nt[nti] + 1
                ncr = Dataset(dirR+'output/FADPrey%s_no%d_npart%d_nfad%d_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc'%(con,iti,npart,nf[nfi],t,f,kp,i,p,pa))
                sFp[nfi,nti,iti,:nf[nfi],:], sFc[nfi,nti,iti,:nf[nfi]+1,:] = FADnt_temp(ncr, nf[nfi])
    ds.close()
