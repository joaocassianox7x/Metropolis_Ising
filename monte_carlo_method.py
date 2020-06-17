import numpy as np
from numpy import random as rand
#import matplotlib.pyplot as plt
#from random import sample
#import numba
#from numba import prange
import time
#from numba import cuda
import multiprocessing as mp

t0=time.time()

def init_conf(l): #apenas numeros pares
    l=int(l)
    if l%2!=0:
        l=l+1 #caso seja ímpar, ele aumenta um l em 1
    n=l*l
    A=[1,-1,1,-1,-1,1,-1,1]
    A=A*int(n/2)
    A=np.array(A)
    A=np.random.choice(A,len(A))
    Aux=np.zeros((l+2,l+2),dtype=np.int)
    
    p=0
    for i in range(1,len(Aux[:,0])-1):
        for j in range(1,len(Aux[:,0])-1):
            Aux[i,j]=A[p]
            p+=1

    return(Aux)

def flip(a,b,B,t=2):
    #a linha do elemento
    #b coluna do elemento
    #A matriz
    #t temperatura
    
    j=np.array([B[a,b-1],B[a-1,b],B[a,b+1],B[a+1,b]])
    i=B[a,b]
    h=1.0 #campo elétrico

    ener=2*i*sum(j)+h
    
    if ener<0:
        i=-1*i
        return i
    elif ener>0 and rand.rand() < np.exp(-ener/t):
        i=-1*i
        return i
    else:
        return i

def magnetism(A):
    return(sum(A)/(len(A[0,:])**2))

def hamiltonian(B,h):
    ener=0
    for a in range(len(B[:,0])):
        for b in range(len(B[:,0])):
            j=np.array([B[a,b-1],B[a-1,b],B[a,b+1],B[a+1,b]])
            ener+=-(np.sum(j)-h)*B[a,b]
    return(ener/4)

dim=4
m0=[]

A=init_conf(dim)
def func(j):
    m=0
    for i in range(dim**2):
        i1,j1=rand.randint(0,dim),rand.randint(0,dim)
        A[i1,j1]=flip(i1, j1, A[:,:])
    m+=np.sum(A)



pool=mp.Pool(mp.cpu_count())
pool=mp.Pool(processes=4)
pool.map(func, range(2,1000,2))
print(time.time()-t0)
    
