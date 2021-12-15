import numpy as np

from parcels import FieldSet, Field
from parcels import ParticleSet
from parcels import Variable
from parcels import ScipyParticle

import time as ostime
import sys
ti0 = ostime.time()

import tunaFADpreykernels as pk

def create_preyfieldRW(lx, ly, res, nprey=100000, Pavg = float(sys.argv[9])):
    # Randomly distribute the prey over the grid
    # Pavg is the average number of prey per grid cell
    dataP = np.zeros(((ly//res)+2, (lx//res)+2))
    gc = (ly/res)*(lx/res) 
    add = 1 * gc / nprey * Pavg
    for n in range(nprey):
        i = np.random.randint(1,dataP.shape[0]-1)
        j = np.random.randint(1,dataP.shape[1]-1)
        dataP[i,j] += add
    assert dataP.max() <= 1
    assert dataP.min() >= 0
    return dataP

def create_preyfieldDG(lx, ly, res, nprey=int(1e5), Pavg = float(sys.argv[9])):
    # Randomly distribute the prey over the grid
    dataP = np.zeros(((ly//res)+2, (lx//res)+2))
    gc = (ly/res)*(lx/res) 
    add = 1 * gc / nprey * Pavg
    for n in range(nprey):
        bo = True
        co = 0
        while(bo):
            co += 1
            i = 1 + np.random.binomial(dataP.shape[0]-2, 0.5)
            j = 1 + np.random.binomial(dataP.shape[1]-2, 0.3)
            if(dataP[i,j]<=1-add):
                dataP[i,j] += add
                bo = False
            elif(co==10):
                bo = False
    # normalize the field
    assert dataP.max() <= 1
    assert dataP.min() >= 0
    return dataP

def create_preyfieldBJ(lx, ly, res, nprey=int(1e5), Pavg = float(sys.argv[9])):
    # Randomly distribute the prey over the grid
    dataP = np.zeros(((ly//res)+2, (lx//res)+2))
    gc = (ly/res)*(lx/res) 
    add = 1 * gc / nprey * Pavg
    for n in range(nprey):
        i = 1 + np.random.binomial(dataP.shape[0]-2, 0.5)
        j = np.random.randint(1,dataP.shape[1]-1)
        dataP[i,j] += add
    assert dataP.max() <= 1
    assert dataP.min() >= 0
    return dataP

if(__name__=='__main__'):
    lx = 140 # habitat length (km)
    ly = int(lx/2) # habitat width (km)
    assert lx%2==0
    assert ly%2==0, 'if prey binomial distribution'
    nfad = int(sys.argv[2]) # number of FADs
    ntuna = int(sys.argv[3]) # number of tuna

    npart = ntuna + nfad + 1 # total number of particles
    seed = int(sys.argv[1])
    np.random.seed(seed) # seeding of Monte Carlo simulations

    # Set the initial locations of the particles
    X = np.random.uniform(0, lx, npart)
    Y = np.random.uniform(0, ly, npart)

    assert (X<=lx).all()
    assert (Y<=ly).all()
    
    # Flow field configuration
    # BJ: Bickley Jet
    # DG: Double Gyre
    # RW: Random Walk
    ff = 'BJ'
    assert ff in ['RW', 'DG', 'BJ']

    # define the particle types: tuna particle is 0, dFAD particle is 1
    ptype = np.zeros(npart)
    # The zeroth particle is only used in fishing strategy FS1.
    # This is article does nothing, but is only located at a random
    # tuna particle before a fishing event, where it acts as a dFAD.
    ptype[:nfad + 1] = 1

    # Define a fieldset without flow
    res = 10 # resolution of the field
    if(ff=='RW'):
        dataP = create_preyfieldRW(lx, ly, res)
    if(ff=='DG'):
        dataP = create_preyfieldDG(lx, ly, res)
    if(ff=='BJ'):
        dataP = create_preyfieldBJ(lx, ly, res)
    gridx, gridy = np.meshgrid(np.arange(-res,lx+res,res), np.arange(-res,ly+res,res))
    gridx = np.array(gridx) + 0.5*res
    gridy = np.array(gridy) + 0.5*res
    fieldset = FieldSet.from_data({'U': np.zeros(dataP.shape), 'V': np.zeros(dataP.shape)},
                                  {'lon': gridx, 'lat': gridy},
                                   mesh='flat')

    # add constant to fieldset, used in FaugerasDiffusion Kernel
    # to determine the strength of displacement due to prey field
    fieldset.add_constant('gres',res)
    fieldset.add_constant('flowtype',ff)
    # Create the field of tuna prey
    assert ly%res==0
    assert lx%res==0
    fieldP = Field('prey', dataP, grid=fieldset.U.grid,
                   interp_method='nearest', mesh='flat')
    fieldset.add_field(fieldP) # prey field added to the velocity FieldSet
    fieldset.prey.to_write = False # enabling the writing of Field prey during execution

    if(nfad>0):
        # Add lists (which are added as an interactive field here)
        # These keep track of the FAD order from FADs with many associated 
        # tuna to FADs with little associated tuna
        # only needed when p>0 in the fishing strategy
        fieldF = Field('FADorders', np.arange(nfad), lon=np.arange(nfad), lat=np.array([0]), time=np.array([0]),
                       interp_method='nearest', mesh='flat', allow_time_extrapolation=True)
        fieldset.add_field(fieldF) # prey field added to the velocity FieldSet
        fieldset.FADorders.to_write = False # enabling the writing of Field prey during execution
        # FAD number where fish is caught
        fieldF = Field('FADc', np.array([0]), lon=np.array([0]), lat=np.array([0]), time=np.array([0]),
                       interp_method='nearest', mesh='flat', allow_time_extrapolation=True)
        fieldset.add_field(fieldF) # prey field added to the velocity FieldSet
        fieldset.FADc.to_write = False # enabling the writing of Field prey during execution

    # list that determines at which tuna particle to fish
    # under fishing strategy FS1
    fieldFe = Field('fe', np.array([0]), lon=np.array([0]), lat=np.array([0]), time=np.array([0]),
                       interp_method='nearest', mesh='flat', allow_time_extrapolation=True)
    fieldset.add_field(fieldFe) # prey field added to the velocity FieldSet
    fieldset.FADc.to_write = False # enabling the writing of Field prey during execution

    # Set the parameters for the model:
    # general
    #  Taxis coefficients
    fieldset.add_constant("kappaT", float(sys.argv[4]))
    fieldset.add_constant("kappaF", float(sys.argv[5]))
    fieldset.add_constant("kappaP", float(sys.argv[6]))
    fieldset.add_constant("kappaI", float(sys.argv[7]))
    max_interaction_distance = 10
    print('realistic FAD-tuna interaction distance is around 10km (7Nm), now (km):',max_interaction_distance)
    fieldset.add_constant("RtF", 2.) # FAD association radius (km)
    fieldset.add_constant("Rtt", 3.) # tuna-tuna max interaction distance (km)
    scale = 300
    fieldset.add_constant("epsP", 12/(24*3600)/scale) # prey depletion by tuna (per second)
    fieldset.add_constant("Td", 2/(24*3600)/scale) # tuna gastric evacuation rate
    fieldset.add_constant("scaleD", scale) # scale tuna gastric evacuation rate

    fieldset.add_constant("epsT", 0.5) # Fraction of associated tuna caught
    p = float(sys.argv[8])
    fieldset.add_constant("p",  p) # p parameter of the geometric distribution
    fieldset.add_constant("nfad", nfad) # total number of FADs
    fieldset.add_constant("ntuna", ntuna) # total number of tuna particles
    # Set a maximum tuna swimming velocity
    fieldset.add_constant("Vmax", (0.4 / 1000)) # km/s
    # the domain
    fieldset.add_constant("Lx", lx)
    fieldset.add_constant("Ly", ly)
    # Determines concentration parameter of von Mises'
    fieldset.add_constant("alpha", 3.) # swimming towards prey
    fieldset.add_constant("gamma", 2.) # swimming towards other tuna

    # Random walk flow:
    if(ff=='RW'):
        fieldset.add_constant_field("Kh_zonalF", 0.05/1000, mesh="flat") # in km/s
        fieldset.add_constant_field("Kh_meridionalF", 0.05/1000, mesh="flat") # in km/s
        fieldset.add_constant_field("Kh_zonalT", 0.1/1000, mesh="flat") 
        fieldset.add_constant_field("Kh_meridionalT", 0.05/1000, mesh="flat") 

    # Parameters of the Logistic curve, which determines
    # the dependence of FAD attraction strength on the number 
    # of associated tuna
    fieldset.add_constant("lL", 1.) # maximum of logistic curve
    fieldset.add_constant("lk", 0.35) # steepness of logistic curve
    fieldset.add_constant("lx0", 12) # value of the sigmoid midpoint
    # And for the vp
    fieldset.add_constant("pL", 2.5) # maximum of logistic curve

    # Parameter for the Bickley Jet flow
    if(ff=='BJ'):
        fieldset.add_constant("Ubj", (.1 / 1000)) # maximum flow strength (km/s)

    # Parameters for the double gyre flow
    if(ff=='DG'):
        fieldset.add_constant("A", (0.05 / 1000)) # flow strength
        fieldset.add_constant("omega", 2*np.pi/ (10*24*60*60)) # frequency of one oscillation (per second)
        fieldset.add_constant("epsDG", 0.2) # 

    # Create custom particle class with extra variable that indicates
    # whether the interaction kernel should be executed on this particle.
    class TFParticle(ScipyParticle):
        ptype = Variable('ptype', dtype=int, to_write='once')
        caught = Variable('caught', dtype=float, initial=0)
        mlon = Variable('mlon', dtype=float, to_write=False, initial=0)
        mlat = Variable('mlat', dtype=float, to_write=False, initial=0)
        FADkap = Variable('FADkap', dtype=float, to_write=True, initial=1.)
        # To govern the displacement of particles (used for velocity normalization):
        dla = Variable('dla', dtype=float, to_write=False, initial=0.)
        dlo = Variable('dlo', dtype=float, to_write=False, initial=0.)
        gradx = Variable('gradx', dtype=float, to_write=False, initial=0.)
        grady = Variable('grady', dtype=float, to_write=False, initial=0.)
        # Stomach fullness:
        St = Variable('St', dtype=float, to_write=False, initial=0.5)
        Sta = Variable('Sta', dtype=float, to_write=True, initial=0)
        Stna = Variable('Stna', dtype=float, to_write=True, initial=0)
        Stac = Variable('Stac', dtype=float, to_write=True, initial=0)
        Stnac = Variable('Stnac', dtype=float, to_write=True, initial=0)

    print('number of FADs: ',np.sum((ptype==1)))
    pset = ParticleSet(fieldset=fieldset, pclass=TFParticle,
                       lon=X, lat=Y,
                       interaction_distance=max_interaction_distance,
                       ptype=ptype)

    output_file = pset.ParticleFile(name="output/FADPrey%s_no%d_npart%d_nfad%d_T%.2f_F%.2f_P%.2f_I%.2f_p%.1f_Pa%.1f.nc"%(ff,seed,npart,nfad,float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7]),float(sys.argv[8]),float(sys.argv[9])),
                                    outputdt=4.32e4) # output twice a day

    rt = 8.64e6 # 100 days of simulation
    print('model run time (days): ',rt/24/3600)

    # set up the kernels, which depends on the configuration used
    kernels = pset.Kernel(pk.CaughtP) + pset.Kernel(pk.GEvacuation) # increase tuna hunger
    ikernels = pset.InteractionKernel(pk.Iattraction)
    if(ff=='DG'):
        kernels += pset.Kernel(pk.DoubleGyre) # Double Gyre flow 
    elif(ff=='RW'):
        kernels += pset.Kernel(pk.DiffusionUniformKhP) # Random walk flow
    if(ff=='BJ'):
        kernels += pset.Kernel(pk.BickleyJet) # Bickley jet flow
        kernels += pset.Kernel(pk.DisplaceParticle) # displace tuna due to swimming
        kernels += pset.Kernel(pk.zper_mrefBC) # reflective boundary conditions
        kernels += pset.Kernel(pk.PreyGrad_zpb) # calculate prey gradient

        ikernels += pset.InteractionKernel(pk.ItunaFAD_zpb)
        ikernels += pset.InteractionKernel(pk.Itunatuna_zpb)
    else:
        kernels += pset.Kernel(pk.DisplaceParticle) # displace tuna due to swimming
        kernels += pset.Kernel(pk.reflectiveBC) # reflective boundary conditions
        kernels += pset.Kernel(pk.PreyGrad) # calculate prey gradient

        ikernels += pset.InteractionKernel(pk.ItunaFAD)
        ikernels += pset.InteractionKernel(pk.Itunatuna)

    kernels += pset.Kernel(pk.FaugerasDiffusion)
    kernels += pset.Kernel(pk.Inertia)
    kernels += pset.Kernel(pk.prevloc)
    kernels += pset.Kernel(pk.PreyDeplete)

    ikernels +=  pset.InteractionKernel(pk.Stcheck)
    if(p!=-2): # p==-2 means that no fish is caught
        ikernels +=  pset.InteractionKernel(pk.ItunaPredFAD)

    pset.execute(pyfunc=kernels,
                 pyfunc_inter=ikernels,
                              # 20 minute time step
                 runtime=rt, dt=1.2e3, output_file=output_file,verbose_progress=False)


    output_file.close()
print('total time (minutes):',(ostime.time()-ti0)/60)
