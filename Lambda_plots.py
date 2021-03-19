# %%
import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy import optimize
from scipy.special import erfinv
import egrss 


n     = 100
sigma = 0.1
a,b   = -0.2, 0.5

x = a + np.sort(np.random.uniform(low = 0, high = 1, size = (n,1)),axis =0) *(b-a)

# Function 
def f(x):
    return 0.25*np.cos(4*np.pi *x) + 0.15*np.sin(12* np.pi *x) + 0.2 * x

# Generate data with noise level sigma
yhat = f(x) + sigma * np.random.normal(loc = 0, scale = 1, size = x.shape)


# Plot of the true function and data 
xx = np.linspace(a,b,300)

f1 = plt.figure()
plt.plot(xx,f(xx),'--',color = 'navy',label = '$y(x)$', linewidth = 2)
plt.plot(x,yhat,'bo',color = 'cornflowerblue',markersize = 5,label = 'observations')
plt.xlim((a,b))
plt.xticks(xx, " ")
plt.yticks(f(xx), " ")
plt.tick_params(left=False, bottom = False, right = False, top = False)
plt.legend(['$y(x)$', 'data'], numpoints = 1, prop = {'size': 16}, loc = 'upper left')
ax = plt.gca()
ax.set_facecolor('whitesmoke')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()
f1.savefig("noisy.pdf", bbox_inches='tight')

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

p      = 2
xi     = (b-a) ** (2 * p - 1)


# Plot estimate for different values of lambda
lam    = 1e-10
Ut, Vt = egrss.generators((x-a)/(b-a),p)
Wt, z  = egrss.potrf(Ut,Vt,n*lam/xi)
c,d    = smoothing_spline_reg(Ut,Wt,z,yhat)
fig,ax = plt.subplots(1,3,figsize=(12, 4))
ax[0].plot(xx,f(xx),'--',color = 'navy', linewidth=1.5)
ax[0].plot(x,yhat,'bo',markersize = 4.5, color = 'cornflowerblue')
ax[0].plot(x,yhat-n*lam/xi*c,color = 'tomato', linewidth=1.2)
ax[0].set_title('$\lambda = {:1.0e}$'.format(lam), fontsize = 18)
ax[0].xaxis.set_visible(False)
ax[0].yaxis.set_visible(False)
ax[0].set_facecolor('whitesmoke')

lam    = 1e-6
Wt, z  = egrss.potrf(Ut,Vt,n*lam/xi)
c,d    = smoothing_spline_reg(Ut,Wt,z,yhat)
ax[1].plot(xx,f(xx),'--',color = 'navy', linewidth=1.5)
ax[1].plot(x,yhat,'bo',markersize = 4.5, color = 'cornflowerblue')
ax[1].plot(x,yhat-n*lam/xi*c,color = 'tomato', linewidth=1.5)
ax[1].set_title('$\lambda = {:1.0e}$'.format(lam), fontsize = 18)
ax[1].xaxis.set_visible(False)
ax[1].yaxis.set_visible(False)
ax[1].set_facecolor('whitesmoke')

lam    = 1e-3
Wt, z  = egrss.potrf(Ut,Vt,n*lam/xi)
c,d    = smoothing_spline_reg(Ut,Wt,z,yhat)
ax[2].plot(xx,f(xx),'--',color = 'navy', linewidth=1.5)
ax[2].plot(x,yhat,'bo',markersize = 4.5, color = 'cornflowerblue')
ax[2].plot(x,yhat-n*lam/xi*c,color = 'tomato', linewidth=1.5)
ax[2].set_title('$\lambda = {:1.0e}$'.format(lam), fontsize = 18)
plt.legend([ '$y(x)$','data','model'],loc = 'upper left', numpoints = 1, prop = {'size': 18}, bbox_to_anchor=(1, 1))
ax[2].xaxis.set_visible(False)
ax[2].yaxis.set_visible(False)
ax[2].set_facecolor('whitesmoke')
plt.show()
fig.savefig("lambda.pdf", bbox_inches='tight')
# %%

# %%
