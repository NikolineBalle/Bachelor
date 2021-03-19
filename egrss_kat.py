# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import factorial

"""
    generators(t, p)
Computes generator representation of p'th order spline kernel matrix generated
by a strictly monotonic vector t of length n.

Ut,Vt = generators(t,p) returns two matrices Ut and Vt of size p-by-n
(with p > 0) such that K = tril(Ut'*Vt) + triu(Vt'*Ut,1) is the kernel
matrix with elements

    K[i,j] = sum_{k=0}^{p-1} (-1)^k/(factorial(p-1-k)*factorial(p+k)
                *(t[i]*t[j])^(p-1-k)*min(t[i],t[j])^(2*k+1)
where t is a nonnegative vector.

# Example
```
p      = 4
t      = np.linspace(1e-2,1,10)
Ut, Vt = generators(t,p)
```
"""
def generators(t,p):
    n       = len(t)
    t.shape = (n,1)
    a       = np.linspace(p-1,0,p)
    a.shape = (p,1)
    b       = np.linspace(p,(2 * p) -1,p)
    b.shape = (p,1)
    Ut = ((np.repeat(t.T,p,0)) ** a) / factorial(a)
    Vt = ((-1) ** a[::-1]) * ( np.repeat(t.T,p,0) ** b) / factorial(b) 
    return Ut, Vt

"""
    full(Ut, Vt[, d])
Forms dense symmetric matrix from generator representation.
K = full(Ut,Vt) forms the symmetric matrix
    K = tril(Ut'*Vt) + triu(Vt'*Ut,1).
    
K = full(Ut,Vt,d) forms the symmetric matrix
    K = tril(Ut'*Vt) + triu(Vt'*Ut,1) + diag(d).
"""
def full(Ut, Vt, d = 'nothing'):
    K = np.tril(np.matmul(Ut.T,Vt),-1)
    K = K + K.T + np.diag(np.sum(Ut * Vt,axis=0).flatten())
    if isinstance(d,str) == False:
        if np.isscalar(d):
            K = K + d* np.eye(Ut.shape[1],Ut.shape[1])#np.diag(d*np.ones(Ut.shape[1]))
        else:
            K = K + np.diag(d.flatten())

    return K


"""
    full_tril(Ut, Wt[, d])
Forms dense lower triangular matrix from generator representation.
L = full_tril(Ut,Wt) forms the lower triangular matrix
    L = tril(Ut'*Wt)
L = full_tril(Ut,Wt,d) forms the lower triangular matrix
    L = tril(Ut'*Wt,-1) + diag(d).
"""    
    
def full_tril(Ut,Wt,d = 'nothing'):  
    if isinstance(d,str):
        return np.tril(np.dot(Ut.T,Wt))
    else:
        if np.isscalar(d):
            c = d * np.ones(Ut.shape[1])
        else:
            c = d.copy()
        return np.tril(np.dot(Ut.T,Wt),-1) + np.diag(c.flatten())


"""
    gemv(Ut, Vt, Pt, Qt, x)
Computes matrix-vector product A*x where A is an extended generator
representable semiseparable matrix given by
   A = tril(Ut'*Vt) + triu(Pt'*Qt,1).
# Example
```
b = gemv(Ut,Vt,Pt,Qt,x)
```
"""
# IMPLEMENT GEMV HERE 


"""
Computes matrix-vector product A*x where A is a symmetric and extended generator
representable semiseparable matrix given by
    A = tril(Ut'*Vt) + triu(Vt'*Ut,1)
# Example
```
b = symv(Ut,Vt,x)
```
"""        
def symv(Ut,Vt,x):
    p,n = Ut.shape
    b   = x.copy() 
    z   = np.zeros(p)
    y   = np.dot(Ut,x)
    for k in range(n):
        z = z + np.dot(Vt[:,k],b[k])
        y = y - np.dot(Ut[:,k],b[k])
        b[k] = np.dot(Ut[:,k].T,z) + np.dot(Vt[:,k].T,y)
    return b

"""
    potrf(Ut, Vt[, d])
Computes implicit Cholesky factorization of the sum of a symmetric extended
generator representable semiseparable matrix and a diagonal matrix.
Wt = potrf(Ut,Vt) computes a matrix Wt such that L = tril(Ut'*Wt) is
the Cholesky factor of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1),
i.e., A = L*L'.

Wt,c = potrf(Ut,Vt,d) computes a matrix Wt and a vector c such that
L = tril(Ut'*Wt,-1) + diag(c) is the Cholesky factor of the symmetric matrix
A = tril(Ut'*Vt) + triu(Vt'*Ut,1) + diag(d), i.e., A = L*L'. The input d must
either be a vector of length size(Ut,2) or a scalar; in the latter case diag(d)
is interpreted as the identity matrix scaled by d.
"""

def potrf(Ut,Vt, d = 'nothing'):
    p,n = Ut.shape
    Wt  = Vt.copy()
    P   = np.zeros((p,p))
    if isinstance(d,str):
        for k in range(n):
            Wt[:,k] = Wt[:,k] - np.dot(P,Ut[:,k])
            Wt[:,k] = Wt[:,k] / np.sqrt(np.dot(Ut[:,k],Wt[:,k]))
            P       = P + np.outer(Wt[:,k],Wt[:,k])
         
        return Wt
    else:
        if np.isscalar(d):
            c = d * np.ones((n,1))
            #print(c)
        else:
            c = d.copy()
            #print(c)
        for k in range(n):
            Wt[:,k] = Wt[:,k] - np.dot(P,Ut[:,k])
            c[k]    = np.sqrt(np.dot(Ut[:,k].T,Wt[:,k])+c[k])
            Wt[:,k] = Wt[:,k] / c[k]
            P       = P + np.outer(Wt[:,k],Wt[:,k])
      
        return Wt, c



"""
    ldl(Ut, Vt[, d])
Computes implicit LDL factorization of the sum of a symmetric extended
generator representable semiseparable matrix and a diagonal matrix.

Wt,c = ldl(Ut,Vt) computes a matrix Wt and a vector c such that the LDL
factorization of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1) + diag(d)
is given by L = tril(Ut'*Wt,-1)+I and D = diag(c), i.e., A = L*D*L'.
                     
Wt,c = ldl(Ut,Vt,d) computes a matrix Wt and a vector c such that the LDL
factorization of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1) is given
by L = tril(Ut'*Wt,-1)+I and D = diag(c), i.e., A = L*D*L'.
"""


def ldl(Ut,Vt, d = 'nothing'):
    p,n = Ut.shape
    Wt  = Vt.copy()
    P   = np.zeros((p,p))
    if isinstance(d,str):
        c = np.zeros((n,1))
        for k in range(n):
            Wt[:,k] = Wt[:,k] - np.dot(P,Ut[:,k])
            c[k]    = np.dot(Ut[:,k].T,Wt[:,k])
            P       = P + np.outer(Wt[:,k],Wt[:,k]) / c[k]
            Wt[:,k] = Wt[:,k] / c[k]
         
    else:
        if np.isscalar(d):
            c = d * np.ones((n,1))
        else:
            c = d.copy()
        for k in range(n):
            Wt[:,k] = Wt[:,k] - np.dot(P,Ut[:,k])
            c[k]    = np.dot(Ut[:,k].T,Wt[:,k]) + c[k]
            P       = P + np.outer(Wt[:,k],Wt[:,k]) / c[k]
            Wt[:,k] = Wt[:,k] / c[k]
    return Wt, c    

"""
Solves the equation L*x = b or L'*x = b where L is a lower triangular extended
generator representable semiseparable or quasi-separable matrix defined in terms
of matrices Ut and Wt, which are of size p-by-n (with p > 0), and in the
quasi-separable case, a vector c of length n.

x = trsv(Ut,Wt,b) solves L*x = b with L = tril(Ut'*Wt).
x = trsv(Ut,Wt,b,'N') is the same as x = trsv(Ut,Wt,b).
x = trsv(Ut,Wt,b,'T') solves L'*x = b with L = tril(Ut'*Wt).
x = trsv(Ut,Wt,c,b) solves L*x = b with L = tril(Ut'*Wt,-1) + diag(c).
x = trsv(Ut,Wt,c,b,'N') is the same as x = trsv(Ut,Wt,c,b).
x = trsv(Ut,Wt,c,b,'T') solves L'*x = b with L = tril(Ut'*Wt,-1) + diag(c).
"""
    
    
def trsv(Ut,Wt,*args):     ## double check the output 
    p,n = Ut.shape
    if  len(args) == 3:
        c = args[0]
        b = args[1]
        trans = args[2]
       
    elif len(args) ==2:
        if isinstance(args[1],str):
            c ='nothing'
            b = args[0]
            trans = args[1]
        else:
            c =args[0]
            b = args[1]
            trans = 'N' 
    elif len(args) ==1:
        c = 'nothing'
        b = args[0]
        trans = 'N'  
    else:
        raise Exception("Invalid number of arguments")

    x = b.copy()
    z = np.zeros((1,p))

    if type(c) == str:
        if trans == 'N':
            for k in range(n):               
                x[k] = (x[k]-np.dot(Ut[:,k],z.T)) / np.dot(Wt[:,k].T,Ut[:,k])             
                z    = z + Wt[:,k] * x[k]
        elif trans == 'T':
            for k in range(n-1,-1,-1): 
                x[k] = (x[k]-np.dot(Wt[:,k],z.T)) / np.dot(Wt[:,k].T,Ut[:,k])
                z    = z + Ut[:,k] * x[k]
        else:
            raise Exception("Expected 'N' or 'T'")
    else:
        if trans == 'N':
            for k in range(n):            
                x[k] = (x[k]-np.dot(Ut[:,k],z.T)) / c[k]
                z    = z + Wt[:,k] * x[k]
        elif trans == 'T':
            for k in range(n-1,-1,-1): 
                x[k] = (x[k]-np.dot(Wt[:,k],z.T)) / c[k]
                z    = z + Ut[:,k] * x[k]
        else:
            raise Exception("Expected 'N' or 'T'")              
    return x

    
    
    
    
    




