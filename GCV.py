# %%
import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy import optimize
from scipy.special import erfinv
from scipy.optimize import minimize_scalar
import egrss as egrss

n     = 100
sigma = 0.1
a,b   = -0.2, 0.5
p = 1  #degree of splines

x = a + np.sort(np.random.uniform(low = 0, high = 1, size = (n,1)),axis =0) *(b-a)

# Function 
def y(x):
    return 0.25*np.cos(4*np.pi *x) + 0.15*np.sin(12* np.pi *x) + 0.2 * x

# Generate data with noise level sigma
yhat = y(x) + sigma * np.random.normal(loc = 0, scale = 1, size = x.shape)


# Plot of the true function and data 
xx = np.linspace(a,b,300)

f1 = plt.figure()
plt.plot(xx,y(xx),'--',color = 'navy',label = '$y(x)$', linewidth = 2)
plt.plot(x,yhat,'bo',color = 'cornflowerblue',markersize = 5,label = 'observations')
plt.xlim((a,b))
plt.xticks(xx, " ")
plt.yticks(y(xx), " ")
plt.tick_params(left=False, bottom = False, right = False, top = False)
plt.legend(['$y(x)$', 'data'], numpoints = 1, prop = {'size': 16}, loc = 'upper left')
ax = plt.gca()
ax.set_facecolor('whitesmoke')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()

def smoothing_spline_reg(Ut,Wt,z,yhat,obj = 'nothing'):
    p,n = Ut.shape
    B   = np.zeros((n,p))
    for k in range(p):
        B[:,k] = egrss.trsv(Ut,Wt,z,Ut[k,:].T,'N')
    Q, R     = qr(B)
    c        = egrss.trsv(Ut,Wt,z,yhat,'N')
    c        = np.dot(Q.T,c)
    d        = solve(R[0:p,0:p],c[0:p])
    c[0:p]   = 0
    c        = egrss.trsv(Ut,Wt,z,np.dot(Q,c),'T')
    if obj == 'nothing':
        return c, d
    elif obj == 'gml': 
        log_glm  = np.log(np.dot(yhat.T,c)) + 2.0 / (n-p ) * (np.sum(np.log(z),axis = 0) + np.sum( np.log(np.abs(np.diag(R[0:p,0:p])))))
        return c, d, log_glm
    else:
        raise Exception("Unknown objective")



def GCV(q):
    a,b   = -0.2, 0.5
    p = 2
    xi     = (b-a) ** (2 * p - 1) 
    lam = 1/np.power(10,q)
    N = 100
    #x = a + np.sort(np.random.uniform(low = 0, high = 1, size = N ),axis =0) *(b-a)
    x = np.linspace(a,b,N)
    yhat = y(x) + sigma * np.random.normal(loc = 0, scale = 1, size = x.shape)

    Ut, Vt = egrss.generators((x-a)/(b-a),p)
    Wt, z  = egrss.potrf(Ut,Vt,N*lam/xi)
    alpha,d    = smoothing_spline_reg(Ut,Wt,z,yhat)
    e = -N*lam/xi*alpha
    W = np.identity(N)
    Linv = egrss.trtri2(Ut,Wt,z)
    trMlam = np.power(np.linalg.norm(Linv, 'fro'),2)
    LinvF = np.zeros([N,p])
    F = np.zeros([N,p])
    for i in range(p):
        F[:,i] = np.power(np.transpose(x),i)

        LinvF[:,i] = egrss.trsv(Ut,Wt,z,F[:,i],'N') 
    Q,R = np.linalg.qr(LinvF, mode = 'reduced')

    LinvT = np.zeros([N,p])
    for i in range(p):
        LinvT[:,i] = egrss.trsv(Ut,Wt,z,Q[:,i],'T') 
    term = np.power(np.linalg.norm(LinvT, 'fro'),2)
    return np.asscalar((e @ W @ e.T)/(N*lam*(trMlam - term)))

res = minimize_scalar(GCV, bounds = (0,10), method='bounded')

plt.figure()
q = np.linspace(0,10,100)
GCV_res = np.zeros(100)
for i in range(0,len(q)):
    GCV_res[i] = GCV(q[i])
plt.plot(q,GCV_res)
plt.plot(res.x, res.fun, 'bo')
plt.show


plt.figure()
a,b   = -0.2, 0.5
p = 2
xi     = (b-a) ** (2 * p - 1) 
lam = 1/np.power(10,res.x)
N = 100
Ut, Vt = egrss.generators((x-a)/(b-a),p)
Wt, z  = egrss.potrf(Ut,Vt,n*lam/xi)
c,d    = smoothing_spline_reg(Ut,Wt,z,yhat)
plt.plot(xx,y(xx),'--',color = 'navy', linewidth=1.5)
plt.plot(x,yhat,'bo',markersize = 4.5, color = 'cornflowerblue')
plt.plot(x,yhat-N*lam/xi*c,color = 'tomato', linewidth=1.2)
plt.xticks(xx, " ")
plt.yticks(y(xx), " ")
plt.tick_params(left=False, bottom = False, right = False, top = False)
plt.legend(['$y(x)$', 'data'], numpoints = 1, prop = {'size': 16}, loc = 'upper left')
ax = plt.gca()
ax.set_facecolor('whitesmoke')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()

# %%
