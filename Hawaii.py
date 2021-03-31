# %%
import numpy as np
import pandas as pd
from numpy.linalg import solve
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy import optimize
from scipy.special import erfinv
from scipy.optimize import minimize_scalar
import egrss 

data = pd.read_csv("archive.csv") 

blue1 = 'cornflowerblue'
blue2 = 'navy'
red = 'tomato'

dates = data['Decimal Date'].to_numpy()
carbon = data['Carbon Dioxide (ppm)'].to_numpy()
dates = dates[np.logical_not(np.isnan(carbon))]
carbon = carbon[np.logical_not(np.isnan(carbon))]

f1 = plt.figure()
plt.plot(dates, carbon, color = blue1)
plt.ylabel('[ppm]', fontsize = 14)
plt.xlabel('[year]', fontsize = 14)
ax = plt.gca()
ax.set_facecolor('whitesmoke')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 12)
plt.title('Carbone dioxide levels in the atmosphere', fontsize = 14)
plt.show()
f1.savefig("Carbonlevels.pdf", bbox_inches='tight')

a = np.min(data['Decimal Date'])
b = np.max(data['Decimal Date'])
N     = carbon.shape[0]
p = 2  #degree of splines
xi     = (b-a) ** (2 * p - 1) 
x = dates
yhat = carbon

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
    
GML_H = minimize_scalar(GML, method='golden')


fig1 = plt.figure()
q = np.linspace(10,-4,30)
GML_H_sim = np.zeros(30)
for i in range(0,len(q)):
    GML_H_sim[i] = GML(q[i])
plt.plot(-q,GML_H_sim, color = 'cornflowerblue')
plt.plot(-GML_H.x, GML_H.fun, 'bo', color = 'navy', label = '$\lambda_{GML}^{*}=0.279$')
ax = plt.gca()
ax.set_facecolor('whitesmoke')
plt.xlabel('1e', loc = 'left', fontsize = 12)
plt.yticks(GML_H_sim, "")
plt.xticks(fontsize = 12)
plt.title('$GML(\lambda)$', fontsize = 16)
plt.tick_params(left=False, bottom = True, right = False, top = False)
ax = plt.gca()
ax.set_facecolor('whitesmoke')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.legend( loc='upper right', numpoints=1, prop={'size': 16} )
plt.show
fig1.savefig("GMLlambdaH.pdf", bbox_inches='tight')

fig2 = plt.figure()
lam = 1/np.power(10,GML_H.x)
Ut, Vt = egrss.generators((x-a)/(b-a),p)
Wt, z  = egrss.potrf(Ut,Vt,N*lam/xi)
c,d    = smoothing_spline_reg(Ut,Wt,z,yhat)
plt.plot(dates,carbon,color = blue1, linewidth=1.5)
plt.plot(x,yhat-N*lam/xi*c,color = red , linewidth=1.2)
plt.tick_params(left=False, bottom = False, right = False, top = False)
plt.legend(['CO2 concentration', 'model' ], numpoints = 1, prop = {'size': 14}, loc = 'best')
plt.ylabel('[ppm]', fontsize = 14)
plt.xlabel('[year]', fontsize = 14)
ax = plt.gca()
ax.set_facecolor('whitesmoke')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 12)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.title('$\lambda_{GML}^{*}=0.279$', fontsize = 18)
plt.show()
fig2.savefig("GMLmodelH.pdf", bbox_inches='tight')

# %%
del lam 
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

GCV_H = minimize_scalar(GCV,  method='golden')


# plotting GCV(lambda)
fig3 = plt.figure()
q = np.linspace(10,-4,30)
GCV_res = np.zeros(30)
for i in range(0,len(q)):
    GCV_res[i] = GCV(q[i])
plt.plot(-q,GCV_res, color = 'cornflowerblue')
plt.plot(-GCV_H.x, GCV_H.fun, 'bo', color = 'navy', label = '$\lambda_{GCV}^{*}=0.068$')
ax = plt.gca()
ax.set_facecolor('whitesmoke')
plt.xlabel('1e', loc = 'left', fontsize = 12)
plt.yticks(GCV_res, "")
plt.xticks(fontsize = 12)
plt.title('$GCV(\lambda)$', fontsize = 16)
plt.tick_params(left=False, bottom = True, right = False, top = False)

ax = plt.gca()
ax.set_facecolor('whitesmoke')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.legend( loc='upper right', numpoints=1, prop={'size': 16} )
plt.show
fig3.savefig("GCVlambdaH.pdf", bbox_inches='tight')


# Plotting the model generated with lambda_GCV
fig4 = plt.figure()
lam = 1/np.power(10,GCV_H.x)
x = np.linspace(a,b,N)
Ut, Vt = egrss.generators((x-a)/(b-a),p)
Wt, z  = egrss.potrf(Ut,Vt,N*lam/xi)
c,d    = smoothing_spline_reg(Ut,Wt,z,yhat)
plt.plot(dates,carbon,color = blue1, linewidth=1.5)
plt.plot(x,yhat-N*lam/xi*c,color = red , linewidth=1.2)
plt.tick_params(left=False, bottom = False, right = False, top = False)
plt.legend(['CO2 concentration', 'model' ], numpoints = 1, prop = {'size': 14}, loc = 'best')
ax = plt.gca()
ax.set_facecolor('whitesmoke')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 12)
plt.ylabel('[ppm]', fontsize = 14)
plt.xlabel('[year]', fontsize = 14)
plt.title('$\lambda_{GCV}^{*}= 0.068$', fontsize = 18)
plt.show()
fig4.savefig("GCVmodelH.pdf", bbox_inches='tight')


# %%
