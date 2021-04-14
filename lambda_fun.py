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
    Q, R  = np.linalg.qr(B, mode = 'reduced') 
    c   = egrss.trsv(Ut,Wt,yhat,z,'N')  # L\yhat
    d   = Q.T@c                         # Q'*(L\yhat)
    c   = egrss.trsv(Ut,Wt,c-Q@d,z,'T') # L'\((I-Q*Q')*(L\yhat))
    d   = solve(R,d)                    # R\(Q'*(L\yhat))
    if obj == 'nothing':
        return c, d
    elif obj == 'gml': 
        log_glm  = np.log(np.dot(yhat.T,c)) + 2.0 / (n-p ) * (np.sum(np.log(z),axis = 0) + np.sum( np.log(np.abs(np.diag(R[0:p,0:p])))))
        return c, d, log_glm
    else:
        raise Exception("Unknown objective")

# Generalized maximum likelihood
def min_GML(N, x, p, yhat, method, usage,  bounds = 'nothing', q = 'nothing'):
    a = np.min(x)
    b = np.max(x)
    xi = (b-a) ** (2 * p - 1) 

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
def min_GCV(N, x, p, yhat, method, usage,  bounds = 'nothing', q = 'nothing'):
    a = np.min(x)
    b = np.max(x)
    xi = (b-a) ** (2 * p - 1) 
     
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

"""
# Fit when pertubating a point
def pertubation(datapoint, yhat, n_fit, quantile):
    
    #Making pertubations and fit
    
    if datapoint == 'left':
        datapoint = 0
    elif datapoint == 'center':
        datapoint = int(len(yhat)/2)
    elif datapoint == 'right':
        datapoint = len(yhat)-1
    allfit = pd.DataFrame(index=range(N),columns=range(n_fit+1))
    val = yhat[datapoint]
    Ut, Vt = egrss.generators((x-a)/(b-a),p)

    for i in range(n_fit):
        yhat[datapoint] += sigma* np.random.normal(loc = 0, scale = 1, size = 1)

        # Finding the optimal lambda - GML
        res_GML = lambda_fun.min_GCV(N, a, b, p, yhat, 'bounded', 'min',  (1,13))
        lam = 1/np.power(10,res_GML.x) # 8.5e-8
        Wt, z  = egrss.potrf(Ut,Vt,N*lam/xi)
        c,d    = lambda_fun.smoothing_spline_reg(Ut,Wt,z,yhat)
        modelfit = yhat-N*lam/xi*c
        modelfitdf = pd.DataFrame(modelfit)
        allfit[i] = modelfitdf
        yhat[datapoint] = val

    yhat[datapoint] = val

    max_val = allfit.quantile(q=1-(1-quantile)/2, axis=1, numeric_only=True)
    min_val = allfit.quantile(q=(1-quantile)/2, axis=1, numeric_only=True)

    return min_val, max_val, datapoint

"""
