import numpy as np
from numpy import random as rand
import time
import multiprocessing as mp

t0=time.time()

def init_conf(l,free): #apenas numeros pares
                       #free->dimensão do seu problema, 1D,2D ou 3D
    l=int(l)
    if l%2!=0:
        l=l+1 #caso seja ímpar, ele aumenta um l em 1
    n=l**free
    if free==1:
        return(np.random.choice(np.array([1,-1]*int(n/2)),n))
    if free==2:
        n=l*l
        return(np.reshape(np.random.choice(np.array([1,-1]*int(n/2)),n),(l,l)))
    if free==3:
        return(np.reshape(np.random.choice(np.array([1,-1]*int(n/2)),n),(l,l,l)))
    
def flip_them_all(A,h,t):
    #len(A)>>>1 para que haja sucesso
    #A matriz
    #t temperatura
    free=sum(np.shape(A))
    
    if free==1:
        for i in range(len(A)):
            i1=rand.randint(1,len(A)-1)
            j=A[i1-1]+A[i1+1]
            ener=2*A[i1]*j+h
            if ener<0:
                A[i1]=-A[i1]
            elif ener>0 and rand.rand()<np.exp(-ener/t):
                A[i1]=-A[i1]
    
        
    if free==2:
        for i in range(int(len(A)**free)):
            i1=np.rand.randint(1,len(A)-1)
            i2=np.rand.randint(1,len(A)-1)
            j=A[i1,i2-1]+A[i1-1,i2]+A[i1,i2+1]+A[i1+1,i2]
            ener=2*A[i1,i2]*j+h
            if ener<0:
                A[i1,i2]*=-1
            elif ener>0 and rand.rand() < np.exp(-ener/t):
                A[i1,i2]*=-1
        
        
    if free==3:
        for i in range(int(len(A)**free)):
            i1=np.rand.randint(1,len(A)-1)
            i2=np.rand.randint(1,len(A)-1)
            i3=np.random.randint(1,len(A)-1)
            
            j=A[i1,i2-1,i3]+A[i1-1,i2,i3]+A[i1,i2+1,i3]+A[i1+1,i2,i3]+A[i1,i2,i3-1]+A[i1+1,i2,i3-1]
            ener=2*A[i1,i2,i3]*j+h
            if ener<0:
                A[i1,i2,i3]*=-1
            elif ener>0 and rand.rand() < np.exp(-ener/t):
                A[i1,i2,i3]*=-1
    return A

def magnetism(A):
    free=sum(np.shape(A))
    return(sum(A)/(len(A)**free))

# def hamiltonian(A,h):
#     free=sum(np.shape(A))
#     ener=0
#     for a in range(len(A[:,0])):
#         for b in range(len(B[:,0])):
#             j=np.array([A[a,b-1],A[a-1,b],A[a,b+1],A[a+1,b]])
#             ener+=-(np.sum(j)-h)*B[a,b]
#     return(ener/4)


def func(l,dim=2,t=0.1,h=0): #func que principal, para chamar o paralelismo
    A=init_conf(l,dim)
    B=flip_them_all(A,h,t)
    return [l,B]

pool=mp.Pool(mp.cpu_count())
pool=mp.Pool(processes=4)
R=pool.map(func, range(2,8,2))

print(time.time()-t0)
