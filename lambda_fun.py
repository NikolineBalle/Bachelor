# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import solve
from scipy.linalg import qr
from scipy import optimize
from scipy.optimize import minimize_scalar
import egrss 


def smoothing_spline_reg(Ut,Wt,z,yhat,obj = 'nothing'):
    p,n = Ut.shape
    B   = np.zeros((n,p))
    for k in range(p):
        B[:,k] = egrss.trsv(Ut,Wt,Ut[k,:].T, z,'N') 
    Q, R     = qr(B)
    c        = egrss.trsv(Ut,Wt,yhat,z,'N')
    c        = np.dot(Q.T,c)
    d        = solve(R[0:p,0:p],c[0:p])
    c[0:p]   = 0
    c        = egrss.trsv(Ut,Wt,np.dot(Q,c), z,'T')
    if obj == 'nothing':
        return c, d
    elif obj == 'gml': 
        log_glm  = np.log(np.dot(yhat.T,c)) + 2.0 / (n-p ) * (np.sum(np.log(z),axis = 0) + np.sum( np.log(np.abs(np.diag(R[0:p,0:p])))))
        return c, d, log_glm
    else:
        raise Exception("Unknown objective")



# Generalized maximum likelihood
def min_GCV(N, a, b, p, yhat, method, usage,  bounds = 'nothing', q = 'nothing'):
    xi = (b-a) ** (2 * p - 1) 
    x = np.linspace(a,b,N)

    def GML(q):
        lam = 1/np.power(10,q)
        Ut, Vt = egrss.generators((x-a)/(b-a),p)
        Wt, z  = egrss.potrf(Ut,Vt,N*lam/xi)
        LinvF = np.zeros([N,p])
        for i in range(p):
            LinvF[:,i] = egrss.trsv(Ut,Wt,Ut[i,:],z,'N') 
        q,R = np.linalg.qr(LinvF)

        alpha ,d  = smoothing_spline_reg(Ut,Wt,z,yhat) 
        return np.asscalar(np.log(np.transpose(yhat) @ alpha) + 2/(N-p)* np.sum(np.log(z)) + 2/(N-p) *np.log(abs(np.linalg.det(R))))

    if usage == 'min':
        min_point = minimize_scalar(GML, bounds = bounds, method=method )
        return min_point
    
    elif usage == 'evaluate':
        res = np.zeros(len(q))
        for i in range(len(q)):
            res[i] = GML(q[i])
        return res


# Generelized cross validation
def min_GCV(N, a, b, p, yhat, method, usage,  bounds = 'nothing', q = 'nothing'):
    xi = (b-a) ** (2 * p - 1) 
    x = np.linspace(a,b,N)

    def GCV(q):
        lam = 1/np.power(10,q)
        Ut, Vt = egrss.generators((x-a)/(b-a),p) 
        Wt, z  = egrss.potrf(Ut,Vt,N*lam/xi)
        alpha ,d  = smoothing_spline_reg(Ut,Wt,z,yhat) 
        Linv = egrss.trtri2(Ut,Wt,z)
        LinvF = np.zeros([N,p])
        for i in range(p):
            LinvF[:,i] = egrss.trsv(Ut,Wt,Ut[i,:], z,'N') 
        Q,R = np.linalg.qr(LinvF, mode = 'reduced')

        LinvTQ = np.zeros([N,p])
        for k in range(p):
            LinvTQ[:,k] = egrss.trsv(Ut,Wt,Q[:,k],c=z,trans='T')

        return np.asscalar(np.log(N) + 2*np.log(np.linalg.norm(alpha))- 2*np.log(np.linalg.norm(Linv,'fro')**2 - np.linalg.norm(LinvTQ,'fro')**2))
    
    if usage == 'min':
        min_point = minimize_scalar(GCV, bounds = bounds, method=method )
        return min_point
    
    elif usage == 'evaluate':
        res = np.zeros(len(q))
        for i in range(len(q)):
            res[i] = GCV(q[i])
        return res

