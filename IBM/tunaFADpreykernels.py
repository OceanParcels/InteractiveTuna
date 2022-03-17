import numpy as np
import math
import  random 
import parcels.rng as ParcelsRandom
import scipy
from scipy.stats import vonmises

__all__ = ['Iattraction', 'ItunaFAD', 'Itunatuna',
           'Stcheck', 'ItunaPredFAD', # particle-particle interaction kernels
           'ItunaFAD_zpb', 'Itunatuna_zpb',# particle-particle interaction 
           # kernels with zonal periodic boundaries
           'PreyDeplete', 'GEvacuation', # particle-field interaction kernel
           'BickleyJet','DoubleGyre', 'DiffusionUniformKhP', # flow kernels
           'prevloc', 'Inertia', 'CaughtP',
           'PreyGrad', 'PreyGrad_zpb',
           'FaugerasDiffusion','DisplaceParticle',
           'zper_mrefBC','reflectiveBC' # boundary conditions
           ]

#%% Interaction kernels -------------------------------------------------------------------------------------------------------------

def Iattraction(particle, fieldset, time, neighbors, mutator):
    """Kernel determines the attraction strength of FADs,
       determined by Logistic function"""
    # The geometric function
    def geometric(x, p=fieldset.p):
        return (1-p)**(x-1)*p

    def f(particle, nom):  # define mutation function for mutator
        particle.FADkap = nom
    # if the FAD attraction strength is determined
    # by the number of associated tuna
    if(particle.ptype==1 and particle.id!=0):
        nom = 0 # keeps track of number of associated tuna
        for n in neighbors:
            if n.ptype==0:
                dist = ((particle.lat-n.lat)**2+(particle.lon-n.lon)**2)**0.5
                if(dist <= fieldset.RtF):
                    nom += 1
        mutator[particle.id].append((f, [nom]))  # add mutation to the mutator
        #particle.FADkap = nom
        fieldset.FADorders.data[0,0,particle.id] = nom
        fieldset.Forders.grid.time[0] = time # updating Field prey time

    # Draw a fishing location near a FAD from the geometric distribution
    if(particle.id==0):#,time>0
        if(fieldset.p==1. or fieldset.nfad==1):
            fieldset.FADc.data[0,0,0] = 1
            fieldset.FADc.grid.time[0] = time # updating Field prey time
        elif(fieldset.p==0):
            fieldset.FADc.data[0,0,0] = np.random.randint(1,fieldset.nfad+1)
            fieldset.FADc.grid.time[0] = time # updating Field prey time
        else:
            probs = geometric(np.arange(fieldset.nfad))
            fieldset.FADc.data[0,0,0] = random.choices(np.arange(fieldset.nfad), probs)[0] + 1
            fieldset.FADc.grid.time[0] = time # updating Field prey time
    return StateCode.Success

def ItunaFAD(particle, fieldset, time, neighbors, mutator):
    '''InterActionKernel that "pulls" all neighbor tuna particles of FADs 
    toward the FAD'''

    distances = []
    na_neighbors = []

    # the swimming
    if(fieldset.kappaF!=0 and particle.ptype==0 and fieldset.nfad>0): # if tuna swims towards FAD
        # Define the Logistic curve
        def LogisticCurve(x, L=fieldset.lL, k=fieldset.lk, x0=fieldset.lx0):
            # x is the number of associated tuna
            res = 1 + L / (1+math.e**(-k*(x-x0)))
            return res

        DS = [0,0]
        for n in neighbors:
            if n.ptype==1 and n.id!=0:
                x_n = np.array([particle.lat, particle.lon, particle.depth]) # n location
                x_p = np.array([n.lat, n.lon, n.depth]) # FAD location
                assert particle.depth==n.depth, 'this kernel is only supported in two dimensions for now'

                dx = x_p - x_n
                norm = np.linalg.norm(dx)
                if(norm>0):
                    DS[0] += dx[0] / norm * LogisticCurve(n.FADkap) 
                    DS[1] += dx[1] / norm * LogisticCurve(n.FADkap)
        if(DS!=[0,0]):
            VP = [0,0,0]
            VP[0] = DS[0] * fieldset.kappaF
            VP[1] = DS[1] * fieldset.kappaF
            d_vec = VP
            def f(particle, dlat, dlon, ddepth):
                particle.dla += dlat
                particle.dlo += dlon
            mutator[particle.id].append((f, d_vec))

    return StateCode.Success

def Stcheck(particle, fieldset, time, neighbors, mutator):
    # updates the integrated stomach emptiness in
    # the cases that associated with FAD or not
    if(particle.ptype==0):
        def Psta(particle, c):  # define mutation function for mutator
            particle.Stac += c
            particle.Sta += particle.St
        def Pstna(particle, c):  # define mutation function for mutator
            particle.Stnac += c
            particle.Stna += particle.St
        bo = True
        for n in neighbors:
            if(n.ptype==1 and n.id!=0 and bo):
                l_n = (particle.lon-n.lon)**2
                if(l_n > (particle.lon-(n.lon+fieldset.Lx))**2):
                    x_p = np.array([n.lat, n.lon + fieldset.Lx, n.depth])
                    l_n = (particle.lon-(n.lon+fieldset.Lx))**2
                elif(l_n > (particle.lon-(n.lon-fieldset.Lx))**2):
                    x_p = np.array([n.lat, n.lon - fieldset.Lx, n.depth])
                    l_n = (particle.lon-(n.lon-fieldset.Lx))**2
                else:
                    x_p = np.array([n.lat, n.lon, n.depth]) # FAD location

                x_n = np.array([particle.lat, particle.lon, particle.depth])
                dx = x_p - x_n
                norm = np.linalg.norm(dx)

                if(norm>0 and norm<fieldset.RtF):
                    bo = False
                    mutator[particle.id].append((Psta, [1]))
        if(bo):
            mutator[particle.id].append((Pstna, [1]))
    return StateCode.Success

def Itunatuna(particle, fieldset, time, neighbors, mutator):
    '''InterActionKernel that "pulls" all neighbor tuna particles of FADs 
    toward the tuna, also includes tuna predation'''
    # the swimming
    if(fieldset.kappaT!=0 and particle.ptype==0): # if tuna swims towards FAD
        DS = [0,0]
        for n in neighbors:
            if n.ptype==0:
                x_n = np.array([particle.lat, particle.lon, particle.depth]) # n location
                x_p = np.array([n.lat, n.lon, n.depth]) # FAD location

                dx = x_p - x_n
                norm = np.linalg.norm(dx)
                if(norm>0 and norm<3):
                    DS[0] += (dx[0] / norm)
                    DS[1] += (dx[1] / norm)

        if(DS!=[0,0]):
            def f(particle, dlat, dlon, ddepth):
                particle.dla += dlat
                particle.dlo += dlon
            vm = [fieldset.gamma * np.linalg.norm(DS), np.arctan2(DS[1], DS[0])]
            angle = ParcelsRandom.vonmisesvariate(vm[1], vm[0])
            VP = [0,0,0]
            VP[0] = np.cos(angle) * fieldset.kappaT
            VP[1] = np.sin(angle) * fieldset.kappaT 
            d_vec = VP

            mutator[particle.id].append((f, d_vec))
    return StateCode.Success

def ItunaPredFAD(particle, fieldset, time, neighbors, mutator):
    """The predation of tuna near a FAD"""
    # The geometric function
    def geometric(x, p=fieldset.p):
        return (1-p)**(x-1)*p

    # keep track of caught tuna (for both FAD and tuna particles)
    def fcF(particle, dc):
        particle.caught += dc
    def fcN(particle, dc):
        particle.caught += dc
    def reset_flo(particle, plon, plat):
        particle.lon = plon
        particle.lat = plat

    # if FAD, consume tuna with a probability
    day = 86400 # if once a day
    if(day%particle.dt!=0):
        print('no fishing taking place!!!!! Set a different dt value.')

    # Set the fishing location if p==-1
    if(np.isclose(time%day,day-particle.dt) and fieldset.p==-1):
        if(particle.id==0):
            fieldset.fe.data[0,0,0] = random.randint(fieldset.nfad+1,fieldset.nfad+1+fieldset.ntuna)
        elif(particle.id==fieldset.fe.data[0,0,0]):
            mutator[0].append((reset_flo,[particle.lon, particle.lat]))

    if(particle.ptype==1 and time>0 and np.isclose(time%day,0)):  # if FAD, consume tuna with a probability
        if(particle.id==0 and fieldset.p==-1):
            prob = 1
        elif(fieldset.nfad==0):
            prob = 0
        elif(particle.id!=0 and fieldset.p==-1):
            prob = 0
        elif(particle.id==0):
            prob = 0
        else:
            fr = fieldset.FADorders.data[0][0].tolist()
            nl = [fr.index(x) for x in sorted(fr, reverse=True)[:fieldset.nfad]]
            ci = fieldset.FADc.data[0,0,0]-1
            if(particle.id==nl[int(ci)]):
                prob = 1
            else:
                prob = 0

        assert prob in [0,1]
        if(prob>0):
            prob *= fieldset.epsT
            for n in neighbors:
                if n.ptype==0:
                    x_p = np.array([particle.lat, particle.lon, particle.depth]) # FAD location
                    x_n = np.array([n.lat, n.lon, n.depth]) # tuna location
                    dist = np.linalg.norm(x_p-x_n)
                    if(dist<fieldset.RtF and (random.random()<prob)):
                        mutator[particle.id].append((fcF,[1]))  # FAD catches tuna particle
                        mutator[n.id].append((fcN, [1]))  # tuna particle is caught

    return StateCode.Success

#%% Interaction kernels with zonally preiodic boundaries -----------------------------------
def ItunaFAD_zpb(particle, fieldset, time, neighbors, mutator):
    '''InterActionKernel that "pulls" all neighbor tuna particles of FADs 
    toward the FAD, also includes tuna predation by the FAD
    Uses zonally periodic boundary conditions in the interaction'''

    distances = []
    na_neighbors = []

    # the swimming
    if(fieldset.kappaF!=0 and particle.ptype==0 and fieldset.nfad>0): # if tuna swims towards FAD
        # Define the Logistic curve
        def LogisticCurve(x, L=fieldset.lL, k=fieldset.lk, x0=fieldset.lx0):
            # x is the number of associated tuna
            res = 1 + L / (1+math.e**(-k*(x-x0)))
            return res

        DS = [0,0]
        for n in neighbors:
            if(n.ptype==1 and n.id!=0):
                l_n = (particle.lon-n.lon)**2
                if(l_n > (particle.lon-(n.lon+fieldset.Lx))**2):
                    x_p = np.array([n.lat, n.lon + fieldset.Lx, n.depth])
                    l_n = (particle.lon-(n.lon+fieldset.Lx))**2
                elif(l_n > (particle.lon-(n.lon-fieldset.Lx))**2):
                    x_p = np.array([n.lat, n.lon - fieldset.Lx, n.depth])
                    l_n = (particle.lon-(n.lon-fieldset.Lx))**2
                else:
                    x_p = np.array([n.lat, n.lon, n.depth]) # FAD location

                x_n = np.array([particle.lat, particle.lon, particle.depth])

                dx = x_p - x_n
                norm = np.linalg.norm(dx)
                if(norm>0):
                    DS[0] += dx[0] / norm * LogisticCurve(n.FADkap)
                    DS[1] += dx[1] / norm * LogisticCurve(n.FADkap)
        if(DS!=[0,0]):
            VP = [0,0,0]
            VP[0] = DS[0] * fieldset.kappaF
            VP[1] = DS[1] * fieldset.kappaF
            d_vec = VP
            def f(particle, dlat, dlon, ddepth):
                particle.dla += dlat
                particle.dlo += dlon
            mutator[particle.id].append((f, d_vec))

    return StateCode.Success

def Itunatuna_zpb(particle, fieldset, time, neighbors, mutator):
    '''InterActionKernel that "pulls" all neighbor tuna particles 
    toward the tuna, also includes tuna predation
    Uses zonally periodic boundary conditions in the interaction'''
    # the swimming
    if(fieldset.kappaT!=0 and particle.ptype==0): # if tuna swims towards FAD
        DS = [0,0]
        for n in neighbors:
            if n.ptype==0:
                l_n = (particle.lon-n.lon)**2
                if(l_n > (particle.lon-(n.lon+fieldset.Lx))**2):
                    x_n = np.array([particle.lat, particle.lon + fieldset.Lx,
                                    particle.depth])
                    l_n = (particle.lon-(n.lon+fieldset.Lx))**2
                if(l_n > (particle.lon-(n.lon-fieldset.Lx))**2):
                    x_n = np.array([particle.lat, particle.lon - fieldset.Lx,
                                    particle.depth])
                    l_n = (particle.lon-(n.lon-fieldset.Lx))**2
                else:
                    x_n = np.array([particle.lat, particle.lon,
                                    particle.depth])
                x_p = np.array([n.lat, n.lon, n.depth])

                dx = x_p - x_n
                norm = np.linalg.norm(dx)
                if(norm>0 and norm<3):
                    DS[0] += (dx[0] / norm)
                    DS[1] += (dx[1] / norm)

        def f(particle, dlat, dlon, ddepth):
            particle.dla += dlat
            particle.dlo += dlon

        if(DS!=[0,0]):
            vm = [fieldset.gamma * np.linalg.norm(DS),
                  np.arctan2(DS[1], DS[0])]
            angle = ParcelsRandom.vonmisesvariate(vm[1], vm[0])
            VP = [0,0,0]
            VP[0] = np.cos(angle) * fieldset.kappaT
            VP[1] = np.sin(angle) * fieldset.kappaT
            d_vec = VP

            mutator[particle.id].append((f, d_vec))
    return StateCode.Success

#%% Field-particle interaction kernels ------------------------------------------------------
def PreyDeplete(particle, fieldset, time):
    if(particle.ptype==0): # if tuna
        xi = (np.abs(np.array(fieldset.prey.lon[0])-particle.lon)).argmin()
        yi = (np.abs(np.array(fieldset.prey.lat[:,0])-particle.lat)).argmin()
        preyno = fieldset.prey.data[0,yi,xi]
        dep = min(fieldset.epsP*particle.dt, preyno)
        if(1-particle.St>fieldset.scaleD*dep):
            # increase stomach fullness
            particle.St += fieldset.scaleD*dep
            # deplete prey from fieldset
            fieldset.prey.data[0,yi,xi] -= dep

            # prey added to field according to some distribution
            bo = True
            for n in range(3):
                if(bo):
                    # Add the depleted prey again at random in the domain
                    if(fieldset.flowtype=='RW'):
                        i = np.random.randint(1,
                                              fieldset.prey.data[0].shape[1]-1)
                        j = np.random.randint(1,
                                              fieldset.prey.data[0].shape[0]-1)
                    elif(fieldset.flowtype=='DG'):
                        i = 1+np.random.binomial(fieldset.prey.data[0].shape[1]-2,
                                                 0.3)
                        j = 1+np.random.binomial(fieldset.prey.data[0].shape[0]-2,
                                                 0.5)
                    else:
                        i = np.random.randint(1,
                                              fieldset.prey.data[0].shape[1]-1)
                        j = 1+np.random.binomial(fieldset.prey.data[0].shape[0]-2,
                                                 0.5)
                    if(fieldset.prey.data[0, j, i]<=(1-dep)):
                        fieldset.prey.data[0, j, i] += dep
                        bo = False
            if(bo):
                print('did not add the depletion properly')

        assert fieldset.prey.data[0].min() >=0 
        assert fieldset.prey.data[0].max() <=1 
        fieldset.prey.grid.time[0] = time # updating Field prey time

# the tuna stomach gets emptier over time
def GEvacuation(particle, fieldset, time):
    if(particle.ptype==0):
        E = fieldset.Td * fieldset.scaleD * particle.dt
        particle.St -= min(particle.St,E)

#%% Advection kernels --------------------------------------------------------------------------
def BickleyJet(particle, fieldset, time):
#Parameters for the Bickley jet
    Ubj = fieldset.Ubj
    L = 1770.
    r0 = 6371.
    k1 = 2 * 1/ r0
    k2 = 2 * 2 / r0
    k3 = 2 * 3/ r0
    eps1 = 0.075
    eps2 = 0.4
    eps3 = 0.3
    c3 = 0.461 * Ubj
    c2 = 0.205 * Ubj
    c1 = 0.1446 * Ubj

    def scale_loc(y, Lx=fieldset.Lx, Ly=fieldset.Ly):
        x0 = y[0] / Lx * np.pi * r0
        y0 = (y[1]-Ly/2) / (Ly/3) * 3000
        return x0, y0

    def v(y,t): #2D velocity
        t *= 1.5 # speed up time evolution
        x1, x2 = scale_loc(y)
        f1 = eps1 * np.exp(-1j *k1 * c1 * t)
        f2 = eps2 * np.exp(-1j *k2 * c2 * t)
        f3 = eps3 * np.exp(-1j *k3 * c3 * t)
        F1 = f1 * np.exp(1j * k1 * x1)
        F2 = f2 * np.exp(1j * k2 * x1)
        F3 = f3 * np.exp(1j * k3 * x1)    
        G = np.real(np.sum([F1,F2,F3]))
        G_x = np.real(np.sum([1j * k1 *F1, 1j * k2 * F2, 1j * k3 * F3]))    
        u =  Ubj / (np.cosh(x2/L)**2)  +  2 * Ubj * np.sinh(x2/L) / (np.cosh(x2/L)**3) *  G
        vd = Ubj * L * (1./np.cosh(x2/L))**2 * G_x
        return [u,vd]

    x = [particle.lon,
        particle.lat] # the scaled particle location

    #Advection of particles using fourth-order Runge-Kutta integration.
    (u1, v1) = v(x,time)

    lon1, lat1 = (x[0]+ u1*.5*particle.dt, x[1] + v1*.5*particle.dt)
    u2 = v([lon1, lat1], time+0.5*particle.dt)[0]
    v2 = v([lon1, lat1], time+0.5*particle.dt)[1]

    (lon2, lat2) = (x[0] + u2*0.5*particle.dt, x[1] + v2*0.5*particle.dt)
    u3 = v([lon2,lat2], time+0.5*particle.dt)[0]
    v3 = v([lon2,lat2], time+0.5*particle.dt)[1]

    lon3, lat3 = (x[0] + u3*particle.dt, x[1] + v3*particle.dt)
    u4 = v([lon3,lat3],time+particle.dt)[0]
    v4 = v([lon3,lat3],time+particle.dt)[1]

    uvel = (u1 + 2*u2 + 2*u3 + u4) / 6.
    vvel = (v1 + 2*v2 + 2*v3 + v4) / 6.

    particle.lon += uvel * particle.dt
    particle.lat += vvel * particle.dt

def DoubleGyre(particle, fieldset, time):
    om = fieldset.omega
    eps = fieldset.epsDG

    # The velocity as described by the double gyre flow
    def f(x,t, eps=eps, om=om):
        f1 = eps*np.sin(om*t)*x[0]**2
        f2 = (1-2*eps*np.sin(om*t))*x[0]
        return f1 + f2

    def dfdx(x,t, eps=eps, om=om):
        f1 = 2*eps*np.sin(om*t)*x[0]
        f2 = (1-2*eps*np.sin(om*t))
        return f1 + f2
    
    def v(x,t, A=fieldset.A):
        # returns the velocity in x- and y-direction
        # location x, time t
        res = [0, 0]
        res[0] = -np.pi*A*np.sin(np.pi*f(x,t))* np.cos(np.pi*x[1])
        res[1] = np.pi*A*np.cos(np.pi*f(x,t)) * np.sin(np.pi*x[1]) * dfdx(x,t)
        return res

    x = [particle.lon/fieldset.Ly,particle.lat/fieldset.Ly] # the scaled particle location
    assert x[0]<=2
    assert x[0]>=0
    assert x[1]<=1
    assert x[1]>=0
    #Advection of particles using fourth-order Runge-Kutta integration.
    (u1, v1) = v(x,time)

    lon1, lat1 = (x[0]+ u1*.5*particle.dt, x[1] + v1*.5*particle.dt)
    u2 = v([lon1, lat1], time+0.5*particle.dt)[0]
    v2 = v([lon1, lat1], time+0.5*particle.dt)[1]

    (lon2, lat2) = (x[0] + u2*0.5*particle.dt, x[1] + v2*0.5*particle.dt)
    u3 = v([lon2,lat2], time+0.5*particle.dt)[0]
    v3 = v([lon2,lat2], time+0.5*particle.dt)[1]

    lon3, lat3 = (x[0] + u3*particle.dt, x[1] + v3*particle.dt)
    u4 = v([lon3,lat3],time+particle.dt)[0]
    v4 = v([lon3,lat3],time+particle.dt)[1]

    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt

def DiffusionUniformKhP(particle, fieldset, time):
    """Same as the DiffusionUniformKh kernel,
    but allows different Kh for different particle types"""
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

    if(particle.ptype==0):
        bx = math.sqrt(2 * fieldset.Kh_zonalT[particle])
        by = math.sqrt(2 * fieldset.Kh_meridionalT[particle])
    elif(particle.ptype==1):
        bx = math.sqrt(2 * fieldset.Kh_zonalF[particle])
        by = math.sqrt(2 * fieldset.Kh_meridionalF[particle])
    else:
        bx = 0
        by = 0
    assert 1e20>particle.lon, 'beginU  %.3f'%(particle.lon)
    particle.lon += bx * dWx
    particle.lat += by * dWy
    assert 1e20>particle.lon, 'endU'

#%% Other kernels --------------------------------------------------------------------------
def prevloc(particle, fieldset, time):
    particle.mlon = particle.lon
    particle.mlat = particle.lat

def Inertia(particle, fieldset, time):
    if(particle.ptype==0 and time>0):
        dlon = particle.lon - particle.mlon
        dlat = particle.lat - particle.mlat
        norm = (dlon**2+dlat**2)**0.5
        if(norm!=0):
            particle.dlo += fieldset.kappaI * dlon / norm
            particle.dla += fieldset.kappaI * dlat / norm
        assert particle.dlo<1e20

def CaughtP(particle, fieldset, time):
    # Reposition the tuna to a random location, if it has been caught
    if(particle.ptype!=1):
        if(particle.caught>0):
            particle.lon = random.uniform(0,fieldset.Lx)
            particle.lat = random.uniform(0,fieldset.Ly)
            particle.caught=0

def PreyGrad(particle, fieldset, time):
    # calculate the prey gradient at particle location
    if(particle.ptype==0):
        if(particle.lon>fieldset.Lx-fieldset.gres/2):
            gradresr = 0
            gradresl = fieldset.gres
        else:
            gradresr = fieldset.gres/2
        if(particle.lon<fieldset.gres/2):
            gradresl = 0
            gradresr = fieldset.gres
        else:
            gradresl = fieldset.gres/2
        if(particle.lat>fieldset.Ly-fieldset.gres/2):
            gradresu = 0
            gradresd = fieldset.gres
        else:
            gradresu = fieldset.gres/2
        if(particle.lat<fieldset.gres/2):
            gradresd = 0
            gradresu = fieldset.gres
        else:
            gradresd = fieldset.gres/2
        particle.gradx = (fieldset.prey[0,0,particle.lat, particle.lon+gradresr] -
                fieldset.prey[0,0,particle.lat, particle.lon-gradresl]) 
        particle.grady = (fieldset.prey[0,0,particle.lat+gradresu, particle.lon] -
                fieldset.prey[0,0,particle.lat-gradresd, particle.lon])

def PreyGrad_zpb(particle, fieldset, time):
    # Caluculate the prey gradient, while taking into account
    # zonally periodic boundaries
    if(particle.ptype==0):
        if(particle.lon>fieldset.Lx-fieldset.gres/2):
            gradresr = 2*(fieldset.Lx-particle.lon) - fieldset.Lx
            gradresl = fieldset.gres
        else:
            gradresr = fieldset.gres/2
        if(particle.lon<fieldset.gres/2):
            gradresl = 2*particle.lon - fieldset.Lx
            gradresr = fieldset.gres
        else:
            gradresl = fieldset.gres/2
        if(particle.lat>fieldset.Ly-fieldset.gres/2):
            gradresu = 0
            gradresd = fieldset.gres
        else:
            gradresu = fieldset.gres/2
        if(particle.lat<fieldset.gres/2):
            gradresd = 0
            gradresu = fieldset.gres
        else:
            gradresd = fieldset.gres/2

        particle.gradx = (fieldset.prey[0,0,particle.lat, particle.lon+gradresr] -
                fieldset.prey[0,0,particle.lat, particle.lon-gradresl]) / fieldset.gres
        particle.grady = (fieldset.prey[0,0,particle.lat+gradresu, particle.lon] -
                fieldset.prey[0,0,particle.lat-gradresd, particle.lon]) / fieldset.gres

def FaugerasDiffusion(particle, fieldset, time):
    # Faugeras diffusion kernel
    if(particle.ptype==0):
        def LogisticCurve(x, L=fieldset.pL, k=15, x0=0.7):
            # x is the stomach fullness
            x = 1 - x # make it stomach emptiness
            res = L / (1+math.e**(-k*(x-x0)))
            return res
        mu = np.arctan2(particle.grady,particle.gradx) # mean angle based on prey field gradient

        kappaM = fieldset.alpha * np.linalg.norm([particle.gradx*fieldset.gres, particle.grady*fieldset.gres]) # standard deviation angle
        angle = ParcelsRandom.vonmisesvariate(mu,kappaM)
 
        # the particle displacement
        particle.dlo += fieldset.kappaP * np.cos(angle) * LogisticCurve(particle.St)
        particle.dla += fieldset.kappaP * np.sin(angle) * LogisticCurve(particle.St)

def DisplaceParticle(particle, fieldset, time):
    # displace the particle according to the summed swimming behaviour of
    # the other particles
    if(particle.ptype==0):
        v = fieldset.Vmax*(1-fieldset.prey[0,0,particle.lat, particle.lon])
        norm = (particle.dlo**2+particle.dla**2)**0.5
        # Displace with correct magnitude:
        if(norm>0):
            particle.dlo = particle.dlo / norm * v
            particle.dla = particle.dla / norm * v
        # displace particle    
        particle.lon += particle.dlo * particle.dt
        particle.lat += particle.dla * particle.dt
        # set the displacement per timestep to zero again
        particle.dlo = 0
        particle.dla = 0

#%% boundary conditions ---------------------------------------------------------------------
# reflective in all directions
def reflectiveBC(particle, fieldset, time):
    if(particle.lat>fieldset.Ly):
        particle.lat = fieldset.Ly - (particle.lat-fieldset.Ly)
    elif(particle.lat<0):
        particle.lat *= -1
    if(particle.lon>fieldset.Lx):
        particle.lon = fieldset.Lx - (particle.lon-fieldset.Lx)
    elif(particle.lon<0):
        particle.lon *= -1

# periodic in zonal direction
# reflective in meridional direction
# used for the bickley jet flow
def zper_mrefBC(particle, fieldset, time):
    if(particle.lat>fieldset.Ly):
        particle.lat = fieldset.Ly - (particle.lat-fieldset.Ly)
    elif(particle.lat<0):
        particle.lat *= -1
    if(particle.lon>fieldset.Lx):
        particle.lon -= fieldset.Lx
    elif(particle.lon<0):
        particle.lon += fieldset.Lx
