
import sys
# This file takes potential depths, relative phase and number of bands req as inputs from the cmd line. 
# A_1 and A_2 are potential depths [in units of energy recoil Er = hbar pi^2/2m where m in the mass of Rubidium 69 ] and can take any value. We consider 0 <= A_1,2 <= 3 to be 'low potential depths' and A_{1,2}>=5 to be 'high potential depths'. High potential depths manifest as exceptionally flat bands (although not mathematically flat!). This regime easily simulates Rabi oscillations between pairs of bands. Low potential depths manifest with curvature in the band structure, very useful for simulating Landau Zener dynamics between pairs of bands.

# phi can take any real value, but the most interesting ones are phi \in [0,pi/8]. Unfortunately for low potential depths phi has no discernable effect. For high potential depths however, it can be used to strongly couple or isolate bands. phi = 0 creates a strongly coupled, (almost degenerate) and *isolated* band multiplet in the second and third bands (first and second excited). Isolated in the sense that the other bands are still present in the system, but there is no interaction with them.

# phi = pi/8 creates two isolated, coupled band multiplets in bands 1,2 and bands 3,4. In this case there is no interaction between the multiplets --- unfortunately these multiplets are not independentally controllable, even though they do not interact with each other. 

# phi = pi/20 creates an >>almost<< equidistant band structure. With appropriately high potential depths there will be no interaction between the individual bands

# The number of bands should just be an integer number. Our wave vector ratio k_1 / k_2 = 5 / 4 means that the lowest 5 / 6 bands are the most interesting.  

A_1 = int(sys.argv[1])
A_2 = int(sys.argv[2])
phi = float(sys.argv[3])
phistring = str(phi).replace('.','-') # 'Needed' for output plot name 
bands = int(sys.argv[4]) 

import numpy as np
from numpy import linalg as la
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
grav = .354692 # Gravity in units of Er
pi = np.pi
e = np.e
truncation = 10 # higher truncation means more of the infinite Hamiltonian is being considered. Better, but slower, simulations
dim = 2*truncation+1 # Dimension of the adjacency of the Hamiltonian
momentum_recoil = pi/4 # Brillouin zone edge
num_of_bands = 10 # Due to eigensystem ordering we usually take a large number of bands...just to be safe...
discret = 50 # Momentum space discretisation
force = grav # As of yet a useless parameter.

dK = 2*momentum_recoil/discret # momentum steps to take
dT = dK/force # time steps to take (useless until I add the Bloch oscillation bit)

# First Brillouin Zone
fbz = np.linspace(0,2*momentum_recoil-dK,discret)

# Superlattice function returns a 2-D list of (x, f(x)) for plotting
def superlattice(a1, a2, phase, xmin, xmax):
    sl = []
    it = 0
    for x in np.linspace(xmin,xmax,500):
        sl.append([x,-a1*np.cos(4*pi*x)**2-a2*np.cos(5*pi*x+phase)**2])
    return([list(x) for x in zip(*sl)])

# =============================================================================
# Define a sparse Hamiltonian and reduced Hamiltonian solely for eigenvalue purposes
# =============================================================================
def Hamiltonian(momenta):
    UpperDiag = np.zeros(truncation)
    LowerDiag = np.zeros(truncation)
    it = 0
    for i in range(0,truncation):
        UpperDiag[it] = (momenta/pi-truncation/2+((i+1)-1)/2)**2-A_1/2-A_2/2
        LowerDiag[it] = (momenta/pi+1/2*(i+1))**2-A_1/2-A_2/2
        it+=1
    Centre = (momenta/pi)**2-(A_1+A_2)/2
    diag = [i for i in range(0,dim)]
    data = np.concatenate((UpperDiag,[Centre],LowerDiag))
    MainDiagonal = sparse.coo_matrix((data, (diag, diag)), shape=(dim,dim)).tocsr()
    FirstLattice = sparse.diags([[-A_1/4]*(dim-4)]*2,[4,-4])
    SecondLattice = sparse.diags([[-A_2/4 * e**(2j*phi)]*(dim-5),[-A_2/4 * e**(-2j*phi)]*(dim-5)],[5,-5])
    return(MainDiagonal+FirstLattice+SecondLattice)
    
def reducedH(momenta):
    return(Hamiltonian(momenta)-np.identity(dim)*10**3)

# =============================================================================
# Eigensystem
# =============================================================================

values = []
vectors = []
for k in fbz:
    w, v = la.eig(np.asarray(reducedH(k)))
    w = np.sort(w.real)
    values.append(w+10**3)
    vectors.append(v)
values = np.asarray(values).T
vectors = np.asarray(vectors)

def plot_energybands(bandstoplot):
    fig = plt.figure()
    for ij in range(0,bandstoplot):
        plt.plot(4*fbz,values[ij])
    plt.title('Lowest %s energy bands'%bandstoplot)
    plt.xlabel('Quasi-momentum: 4kd')
    plt.ylabel('Energy: E/E_R')
    plt.savefig("EnergyBands%s_A1_%s_A2_%s_phi_%s.png"%(bands,A_1,A_2,phistring))  
    
def plot_superlattice(a1,a2,phase,xmin,xmax):
    fig=plt.figure()
    superlatticelist=superlattice(a1,a2,phase,xmin,xmax)
    plt.plot(superlatticelist[0],superlatticelist[1])
    plt.title('Superlattice in real space')
    plt.xlabel('Real space: x/4d')
    plt.ylabel('Energy: V(x)/E_R')
    plt.savefig("Superlattice_A1_%s_A2_%s_phi_%s.png"%(A_1,A_2,phistring))  
    
plot_superlattice(A_1,A_2,phi,0,4)
plot_energybands(bands)
