
import sys

A_1=int(sys.argv[1])
A_2=int(sys.argv[2])
phi=float(sys.argv[3])
phistring=str(phi).replace('.','-')
bands=int(sys.argv[4])

import numpy as np
from numpy import linalg as la
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
grav=.354692
pi=np.pi
e=np.e
truncation=10
dim=2*truncation+1
momentum_recoil=pi/4
num_of_bands=10
discret=50
force=grav

dK=2*momentum_recoil/discret
dT=dK/force
#First Brillouin Zone
fbz=np.linspace(0,2*momentum_recoil-dK,discret)

def superlattice(a1,a2,phase,xmin,xmax):
    sl=[]
    it=0
    for x in np.linspace(xmin,xmax,500):
        sl.append([x,-a1*np.cos(4*pi*x)**2-a2*np.cos(5*pi*x+phase)**2])
    return([list(x) for x in zip(*sl)])

# =============================================================================
# Define a sparse Hamiltonian and reduced Hamiltonian solely for eigenvalue purposes
# =============================================================================
def Hamiltonian(momenta):
    UpperDiag=np.zeros(truncation)
    LowerDiag=np.zeros(truncation)
    it=0
    for i in range(0,truncation):
        UpperDiag[it]=(momenta/pi-truncation/2+((i+1)-1)/2)**2-A_1/2-A_2/2
        LowerDiag[it]=(momenta/pi+1/2*(i+1))**2-A_1/2-A_2/2
        it+=1
    Centre=(momenta/pi)**2-(A_1+A_2)/2
    diag=[i for i in range(0,dim)]
    data=np.concatenate((UpperDiag,[Centre],LowerDiag))
    MainDiagonal=sparse.coo_matrix((data, (diag, diag)), shape=(dim,dim)).tocsr()
    FirstLattice=sparse.diags([[-A_1/4]*(dim-4)]*2,[4,-4])
    SecondLattice=sparse.diags([[-A_2/4 * e**(2j*phi)]*(dim-5),[-A_2/4 * e**(-2j*phi)]*(dim-5)],[5,-5])
    return(MainDiagonal+FirstLattice+SecondLattice)
    
def reducedH(momenta):
    return(Hamiltonian(momenta)-np.identity(dim)*10**3)

# =============================================================================
# Eigensystem
# =============================================================================

values=[]
vectors=[]
for k in fbz:
    w,v=la.eig(np.asarray(reducedH(k)))
    w=np.sort(w.real)
    values.append(w+10**3)
    vectors.append(v)
values=np.asarray(values).T
vectors=np.asarray(vectors)

def plot_energybands(bandstoplot):
    fig=plt.figure()
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
