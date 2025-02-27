import numpy as np
import scipy as sp
import scipy.io	 # NOQA
import pylab as pl
from pylab import rcParams

def preProcess(p):
	# omits prob 0 states
	# if pX[ind]=0 then we want to get rid of those indices
	p = p[np.nonzero(np.sum(p,1)),:][0]
	p = p[:,np.nonzero(np.sum(p,0))]
	p = p[:,0,:]

	# recalculate marginals
	pX = np.sum(p,1)
	pYgX = np.dot(np.diag(1/pX),p)

	# cluster nearly equivalent pYgX
	i = 0
	while i<p.shape[0]:
		foo = pYgX[i,:]
		todelete = []
		for j in xrange(i+1,p.shape[0]):
			foo2 = pYgX[j,:]
			if np.max(np.abs(foo-foo2))<1e-15:
				todelete.append(j)
				p[i,:] += p[j,:]
			else:
				pass
		p = np.delete(p,todelete,axis=0)
		pX = np.sum(p,1)
		pYgX = np.dot(np.diag(1/pX),p)
		i = i+1

	# recalculate marginals
	pY = np.sum(p,0)
	pXgY = np.dot(p,np.diag(1/pY))

	i = 0
	while i<p.shape[1]:
		foo = pXgY[:,i]
		todelete = []
		for j in xrange(i+1,p.shape[1]):
			foo2 = pXgY[:,j]
			if np.max(np.abs(foo-foo2))<1e-15:
				todelete.append(j)
				p[:,i] += p[:,j]
			else:
				pass
		p = np.delete(p,todelete,axis=1)
		pY = np.sum(p,0)
		pXgY = np.dot(p,np.diag(1/pY))
		i = i+1

	# recalculate marginals
	pX = np.sum(p,1)
	pYgX = np.dot(np.diag(1/pX),p)

	m = len(pX)
	d = np.zeros([m,m])
	for i in xrange(m):
		for j in xrange(m):
			d[j,i]=np.nansum(pYgX[i,:]*np.log2(pYgX[i,:]/pYgX[j,:]))

	# if there are infinities in d, make sure to avoid unstable fixed points
	if np.sum(np.isinf(d))>0:
		p += 1e-12*np.random.uniform(size=p.shape)
		p /= np.sum(p)
	else:
		pass

	return p

def getFeatures(p,beta,pRgX0):
	# take in joint prob dist p and return p(R|X) at beta
	# pRgX0 is some initial guess for the pRgX at that beta

	# calculate marginals
	pX = np.sum(p,1)
	pY = np.sum(p,0)
	pYgX = np.dot(np.diag(1/pX),p)

	# 2. initialize codebook matrices
	pmin = np.min(pRgX0[pRgX0>0])
	pRgX = pRgX0+1e-9*pmin*np.random.uniform(size=pRgX0.shape) # add noise
	pRgX = np.fmax(pRgX,0*pRgX) # put floor on prob. being 0
	pRgX = np.dot(pRgX,np.diag(1/np.sum(pRgX,0)))
	pR = np.dot(pRgX,pX) 
	pXgR = np.dot(np.diag(1/pR),np.dot(pRgX,np.diag(pX)))
	pYgR = np.dot(pXgR,pYgX)

	m = len(pX)
	d = np.zeros([m,m])
	for i in xrange(m):
		for j in xrange(m):
			d[j,i]=np.nansum(pYgX[i,:]*np.log2(pYgX[i,:]/pYgR[j,:]))

	for t in range(0,300): # arbitrary, fix this
		# run dynamical system
		pRgX = np.dot(np.diag(pR),np.exp(-beta*d))
		Z = np.sum(pRgX,0)
		pRgX = np.dot(pRgX,np.diag(1/Z))
		pR = np.dot(pRgX,pX)
		# calculate new distortion matrix
		pXgR = np.dot(np.diag(1/pR),np.dot(pRgX,np.diag(pX)))
		pYgR = np.dot(pXgR,pYgX)
		for i in xrange(m):
			for j in xrange(m):
				d[j,i]=np.nansum(pYgX[i,:]*np.log2(pYgX[i,:]/pYgR[j,:]))

	R = -np.nansum(pR*np.log2(pR))+np.dot(pX,np.nansum(pRgX*np.log2(pRgX),0)) # I[R;X]
	D = -np.nansum(pY*np.log2(pY))+np.dot(pR,np.nansum(pYgR*np.log2(pYgR),1)) # I[R;Y]

	return pRgX, pXgR, R, D

def getFeatures2(p,beta,pRgX0):
	# take in joint prob dist p and return p(R|X) at beta
	# pRgX0 is some initial guess for the pRgX at that beta

	# calculate marginals
	pX = np.sum(p,1)
	pY = np.sum(p,0)
	pYgX = np.dot(np.diag(1/pX),p)

	# 2. initialize codebook matrices
	uRgX = np.log(pRgX0)-np.log(np.meshgrid(np.sum(pRgX0,0),np.sum(pRgX0,0))[0])
	pRgX = np.exp(uRgX) #np.dot(pRgX,np.diag(1/np.sum(pRgX,0)))
	pR = np.dot(pRgX,pX)
	uXgR = -np.log(np.meshgrid(pR,pR)[1])+np.log(pRgX)+np.log(np.meshgrid(pX,pX)[0])
	pXgR = np.exp(uXgR) #np.dot(np.diag(1/pR),np.dot(pRgX,np.diag(pX)))
	pYgR = np.dot(pXgR,pYgX)

	m = len(pX)
	d = np.zeros([m,m])
	for i in xrange(m):
		for j in xrange(m):
			d[j,i]=np.nansum(pYgX[i,:]*np.log2(pYgX[i,:]/pYgR[j,:]))

	for t in range(0,500): # arbitrary, fix this
		# run dynamical system
		uRgX = np.log(np.meshgrid(pR,pR)[1])-beta*d
		pRgX = np.exp(uRgX) #np.dot(np.diag(pR),np.exp(-beta*d))
		Z = np.sum(pRgX,0)
		uRgX = np.log(pRgX)-np.log(np.meshgrid(Z,Z)[0])
		pRgX = np.exp(uRgX) #np.dot(pRgX,np.diag(1/Z))
		pR = np.dot(pRgX,pX)
		# calculate new distortion matrix
		uXgR = -np.log(np.meshgrid(pR,pR)[1])+np.log(pRgX)+np.log(np.meshgrid(pX,pX)[0])
		pXgR = np.exp(uXgR) #np.dot(np.diag(1/pR),np.dot(pRgX,np.diag(pX)))
		pYgR = np.dot(pXgR,pYgX)
		for i in xrange(m):
			for j in xrange(m):
				d[j,i]=np.nansum(pYgX[i,:]*np.log2(pYgX[i,:]/pYgR[j,:]))

	R = -np.nansum(pR*np.log2(pR))+np.dot(pX,np.nansum(pRgX*np.log2(pRgX),0)) # I[R;X]
	D = -np.nansum(pY*np.log2(pY))+np.dot(pR,np.nansum(pYgR*np.log2(pYgR),1)) # I[R;Y]

	return pRgX, pXgR, R, D

def getFeatures3(p,beta):
	# take in joint prob dist p and beta
	# returns best pRgX it finds
	m, n = p.shape
	pRgX = np.identity(m)
	pRgX, pXgR, r, d = getFeatures2(p,beta,np.identity(m)) # smooth homotopy method
	# sometimes it fails!!
	if d-(1/beta)*r<0:
		# this means that you'd do better to code everything as iid.
		r = 0; d = 0
	else:
		pass
	# do several random restarts
	for k in xrange(1000): # 11 random restarts
		pRgX2, pXgR2, r2, d2 = getFeatures2(p,beta,np.random.uniform(size=pRgX.shape))
		if d2-(1/beta)*r2>d-(1/beta)*r: # take the maximal objective function
			r = r2; d = d2; pRgX = pRgX2
		else:
			pass
	return pRgX, pXgR, r, d

def checkBeta(p,beta,pRgX):
	# calculate marginals
	pX = np.sum(p,1)
	pY = np.sum(p,0)
	pYgX = np.dot(np.diag(1/pX),p)
	# 2. initialize codebook matrices
	pR = np.dot(pRgX,pX)
	uXgR = -np.log(np.meshgrid(pR,pR)[1])+np.log(pRgX)+np.log(np.meshgrid(pX,pX)[0])
	pXgR = np.exp(uXgR) #np.dot(np.diag(1/pR),np.dot(pRgX,np.diag(pX)))
	pYgR = np.dot(pXgR,pYgX)
	r1 = -np.nansum(pR*np.log2(pR))+np.dot(pX,np.nansum(pRgX*np.log2(pRgX),0)) # I[R;X]
	d1 = -np.nansum(pY*np.log2(pY))+np.dot(pR,np.nansum(pYgR*np.log2(pYgR),1)) # I[R;Y]
	# then move beta a teeny bit
	beta += 1e-4
	# find new pRgX with that as initial condition
	pRgX2, pXgR, r2, d2 = getFeatures2(p,beta,pRgX)
	return beta*(r2-r1)/(d2-d1)

def IB_method2(p,beta):
	# take in joint prob dist p and returns (R,D) for each b in beta
	# beta must be in descending order	# initialize R, P arrays
	R = np.zeros(beta.shape)
	D = np.zeros(beta.shape)
	# smooth homotopy method: use previous solution as initial condition to the next
	m, n = p.shape
	pRgX = np.identity(m)
	for i in np.arange(len(beta)):
		pRgX, pXgR, r, d = getFeatures2(p,beta[i],pRgX) # smooth homotopy method
		if d-(1/beta[i])*r<0:
			# this means that you'd do better to code everything as iid.
			r = 0; d = 0
		else:
			pass
		# do several random restarts
		for k in xrange(5000): # 11 random restarts
			if k==0:
				pRgX2, pXgR2, r2, d2 = getFeatures2(p,beta[i],np.identity(m))
			else:
				pRgX2, pXgR2, r2, d2 = getFeatures2(p,beta[i],np.random.uniform(size=pRgX.shape))
			if ~np.isnan(r2) and d2-(1/beta[i])*r2>d-(1/beta[i])*r: # take the maximal objective function
				r = r2; d = d2; pRgX = pRgX2
			else:
				pass
		# then, add r and d to the right thing
		R[i] = r
		D[i] = d
		print i
	return R, D

def IB_method(p,beta):
	# take in joint prob dist p and returns (R,D) for each b in beta
	# beta must be in descending order!!!
	# initialize R, P arrays
	R = np.zeros(beta.shape)
	D = np.zeros(beta.shape)
	# smooth homotopy method: use previous solution as initial condition to the next
	m, n = p.shape
	pRgX = np.identity(m)
	for i in np.arange(len(beta)):
		pRgX, pXgR, r, d = getFeatures2(p,beta[i],pRgX) # smooth homotopy method
		# sometimes it fails!!
		while np.isnan(r):
			pRgX, pXgR, r, d = getFeatures2(p,beta[i],np.identity(m))
		# then, add r and d to the right thing
		R[i] = r
		D[i] = d
	return R, D

def TrajDist(eM_name,time_dir,params,m,n):
	# pasts of length m, futures of length n
	# find T0, T1 from eM_name and params
	if eM_name=='Even':
		p = params
		T0 = np.zeros([2,2]); T0[0,0] = p
		T1 = np.zeros([2,2]); T1[1,0] = 1-p; T1[0,1] = 1
	elif eM_name=='RIP':
		p = params[0]
		q = params[1]
		T0 = np.zeros([3,3]); T0[1,0] = p; T0[2,1] = q
		T1 = np.zeros([3,3]); T1[0,2] = 1; T1[2,0] = 1-p; T1[2,1] = 1-q
	elif eM_name=='GoldenMean':
		p = params
		T0 = np.zeros([2,2]); T0[1,0] = 1-p
		T1 = np.zeros([2,2]); T1[0,0] = p; T1[0,1] = 1
	elif eM_name=='SNS': # nonunifilar presentation
		p = params[0]
		q = params[1]
		T0 = np.zeros([2,2]); T0[0,0] = p; T0[1,0] = 1-p; T0[1,1] = q
		T1 = np.zeros([2,2]); T1[0,1] = 1-q
	elif eM_name=='TentMap':
		a=1.7692935
		T0 = np.zeros((4,4))
		T1 = np.zeros((4,4))
		T0[0,2]=1/(a+2)
		T1[0,1]=a/(2*a+2)
		T1[1,0]=1
		T0[2,1]=(a+2)/(2*a+2)
		T0[2,3]=(a**2+2*a)/(2*(a**2)+4*a+2)
		T1[3,2]=(a+1)/(a+2)
		T1[3,3]=((a**2)+2*a+2)/(2*(a**2)+4*a+2)
	elif eM_name=='Butterfly':
		T0 = np.zeros([2,2])
		T1 = np.zeros([2,2])
		T2 = np.zeros([2,2])
		T0[0,0] = 1.0/3.0
		T1[0,0] = 1.0/3.0
		T2[1,0] = 1.0/3.0
		T1[1,1] = 0.5
		T0[0,1] = 0.5
	elif eM_name=='Renewal':
		F = params # F(n)
		F /= np.sum(F) # normalize just in case
		L = len(F)
		w = 1-np.cumsum(F)
		T0 = np.diag(w[1:]/w[:-1],k=-1)
		T1 = np.zeros([L,L])
		T1[0,:] = F/w
	else:
		Stpm = []
	# return matrix p: rows are pasts, columns are futures, entries joint probabilities
	T = T0+T1
	if eM_name=='Butterfly':
		T = T0+T1+T2
	# find the eigenvectors for T
	w, v = np.linalg.eig(T)
	# look for the stationary distribution, w = 1
	PI = v[:,np.abs(w-1)<1e-15]
	PI = PI/np.sum(PI)
	# from PI, generate joint distribution of P(m-length sequence,n-length sequence)
	if eM_name=='Butterfly': # trinary alphabet
		p = np.zeros((pow(3,m),pow(3,n)))
		for i in xrange(pow(3,m)):
			v=PI
			if i==0:
				padi = m-len(np.base_repr(i,base=3))+1
			else:
				padi = m-len(np.base_repr(i,base=3))
			for k in np.base_repr(i,base=3,padding=padi):
				if k=='0':
					v=np.dot(T0,v)
				elif k=='1':
					v=np.dot(T1,v)
				else:
					v=np.dot(T2,v)
			for j in xrange(pow(3,n)):
				v2=v
				padj = n-len(np.base_repr(j,base=3))
				if j==0:
					padj = n-len(np.base_repr(j,base=3))+1
				else:
					padj = n-len(np.base_repr(j,base=3))
				for k in np.base_repr(j,base=3,padding=padj):
					if k=='0':
						v2=np.dot(T0,v2)
					elif k=='1':
						v2=np.dot(T1,v2)
					else:
						v2=np.dot(T2,v2)
				p[i,j] = np.sum(v2)
	else: # binary alphabet
		p = np.zeros((pow(2,m),pow(2,n)))
		for i in xrange(pow(2,m)):
			v=PI
			for k in np.binary_repr(i,m):
				if k=='0':
					v=np.dot(T0,v)
				else:
					v=np.dot(T1,v)

			for j in xrange(pow(2,n)):
				v2=v
				for k in np.binary_repr(j,n):
					if k=='0':
						v2=np.dot(T0,v2)
					else:
						v2=np.dot(T1,v2)
				p[i,j] = np.sum(v2)

	if time_dir=='R':
		p = p.T
	else:
		pass

	return p

def JointCsDist(eM_name,time_dir,params):
	# find forward and reverse time causal states
	# INPUT THE ACTUAL CODE
	if eM_name=='Even':
		p = params
		Stpm = np.identity(2)
		Stpm[0,0] = (1-p)/(2-p)
		Stpm[1,1] = 1/(2-p)
	elif eM_name=='RIP':
		p = params[0]
		q = params[1]
		Stpm = np.zeros([3,4])
		Stpm[0,1] = (1-p)/(p+2)
		Stpm[0,3] = p/(p+2)
		Stpm[1,1] = p*(1-q)/(p+2)
		Stpm[1,2] = p*q/(p+2)
		Stpm[2,0] = 1/(p+2)
	elif eM_name=='GoldenMean':
		p = params
		Stpm = np.zeros([2,2])
		Stpm[0,0] = p/(2-p)
		Stpm[0,1] = (1-p)/(2-p)
		Stpm[1,0] = (1-p)/(2-p)
	elif eM_name=='SNS':
		p = params[0]
		q = params[1]
		L = params[2]
		row, col = np.indices([1000,1000])
		if p!=q:
			Stpm1 = (1-p)*(1-q)*(np.power(p,row+col)-np.power(q,row+col))/(p-q)
			Stpm1 /= np.sum(Stpm1)
		else:
			Stpm1 = (1-p)**2*(row+col)*np.power(p,row+col-1)
			Stpm1 /= np.sum(Stpm1)
		# simplify this to 10 by 10, that's all you really need
		Stpm = np.zeros([L,L])
		Stpm[:L-1,:L-1] = Stpm1[:L-1,:L-1]
		Stpm[L-1,:L-1] = np.sum(Stpm1[L-1:,:L-1],0)
		Stpm[:L-1,L-1] = np.sum(Stpm1[:L-1,L-1:],1)
		Stpm[L-1,L-1] = np.sum(Stpm1[L-1:,L-1:])
	elif eM_name=='TentMap':
		a=1.7692935
		Stpm=np.zeros((4,3))
		Stpm[0,1]=a+1
		Stpm[1,2]=a
		Stpm[2,1]=a+(a**2/2)
		Stpm[3,2]=a+a**2
		Stpm[3,1]=1+a+(a**2/2)
		Stpm[3,0]=1+a
		Stpm[2,2]=a+2
		Stpm[1,0]=1
		Stpm /= np.sum(Stpm)
	elif eM_name=='Butterfly':
		L = params # total length into future or past
		Stpm = np.zeros([2,L+1])
		piR = (1+np.power(2.0/3.0,np.arange(L+1)))/(5*np.power(2,np.arange(L+1)))
		piR[L] = 0.3
		piAgR = 3.0*np.power(2.0,np.arange(1,L+2))/(3*np.power(2,np.arange(1,L+2))+2*np.power(3,np.arange(1,L+2)))
		piAgR[L] = 1
		Stpm[0,:] = piAgR*piR
		Stpm[1,:] = (1-piAgR)*piR
		Stpm /= np.sum(Stpm)
	elif eM_name=='Renewal':
		F = params # F(n)
		F /= np.sum(F) # normalize just in case
		L = len(F)
		row, col = np.indices((L,L))
		mask = np.fmin(row+col,L)
		F = np.hstack([F,0])
		Stpm = F[mask]
		Stpm /= np.sum(Stpm)
	else:
		Stpm = []

	if time_dir=='R':
		Stpm = Stpm.T
	else:
		pass

	return Stpm

def OCF(eM_name,time_dir,params,m,n,betamin,betamax,numsteps):
	beta = np.linspace(betamax,betamin,numsteps) # MUST BE DESCENDING ORDER
	#beta = np.power(10,np.linspace(np.log10(betamax),np.log10(betamin),numsteps)) #semilog plot
	# get trajectory distribution
	p = TrajDist(eM_name,time_dir,params,m,n)
	p = preProcess(p)
	R, D = IB_method2(p,beta)
	return beta, R, D

def CN(eM_name,time_dir,params,betamin,betamax,numsteps):
	beta = np.linspace(betamax,betamin,numsteps) # MUST BE DESCENDING ORDER
	#beta = np.power(10,np.linspace(np.log10(betamax),np.log10(betamin),numsteps)) #semilog plot
	p = JointCsDist(eM_name,time_dir,params)
	p = preProcess(p)
	R, D = IB_method2(p,beta)
	return beta, R, D

################################

# the following stolen from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
fig_width_pt = 2*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)+1.0)/2.0 - 1     # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = .8*fig_width      # height in inches
fig_size =  [fig_width, fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 18,
          'text.fontsize': 18,
          'legend.fontsize': 18,
          'legend.handlelength': 3,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'text.usetex': True,
          'figure.figsize': fig_size}
rcParams.update(params)
# end of stolen content.

def makeFigs(eM_name,time_dir,params,betamax,numsteps):
	# ########
	btrue, Rtrue, Ptrue = CN(eM_name,time_dir,params,0.1,betamax,numsteps)
	b2, R2, P2 = OCF(eM_name,time_dir,params,2,2,0.1,betamax,numsteps)
	b3, R3, P3 = OCF(eM_name,time_dir,params,3,3,0.1,betamax,numsteps)
	b4, R4, P4 = OCF(eM_name,time_dir,params,4,4,0.1,betamax,numsteps)
	b5, R5, P5 = OCF(eM_name,time_dir,params,5,5,0.1,betamax,numsteps)

	Rmax = np.max([Rtrue,R2,R3,R4,R5])
	Cmu = np.max(Rtrue)
	Pmax = np.max(Ptrue[~np.isnan(Ptrue)])
	E = Pmax
	betamax = np.max(btrue)

	# PIC figure
	fig = pl.figure()
	pl.plot(Rtrue, Ptrue, '-o', linewidth=2)
	pl.plot(R2, P2, '--x', linewidth=2)
	pl.plot(R3, P3, '--x', linewidth=2)
	pl.plot(R4, P4, '--x', linewidth=2)
	pl.plot(R5, P5, '--x', linewidth=2)
	pl.plot(Cmu+0*Rtrue, Ptrue, '--k', linewidth=1.5)
	pl.plot(Rtrue, E+0*Ptrue, '--k', linewidth=1.5)
	pl.xlim([0,1.05*Rmax])
	pl.xlabel('$I[\mathcal{R};\overleftarrow{X}]$')
	pl.ylim([0,1.05*Pmax])
	pl.ylabel('$I[\mathcal{R};\overrightarrow{X}]$')
	xlocs = pl.xticks()[0]
	xlabels = np.append(['$'+str(item)+'$' for item in xlocs],'$C_{\mu}$')
	xlocs = np.append(xlocs,Cmu)
	pl.xticks(xlocs,xlabels)
	ylocs = pl.yticks()[0]
	ylabels = np.append(['$'+str(item)+'$' for item in ylocs],'$\mathbf{E}$')
	ylocs = np.append(ylocs,E)
	pl.yticks(ylocs,ylabels)
	pl.savefig(eM_name+time_dir+str(params)+'_PIC.pdf',bbox_inches='tight')
	#pl.show()
	pl.clf()

	# PRD figure
	#fig = pl.figure()
	pl.plot(Pmax-Ptrue, Rtrue, '-o', linewidth=2)
	pl.plot(Pmax-P2, R2, '--x', linewidth=2)
	pl.plot(Pmax-P3, R3, '--x', linewidth=2)
	pl.plot(Pmax-P4, R4, '--x', linewidth=2)
	pl.plot(Pmax-P5, R5, '--x', linewidth=2)
	pl.plot(Ptrue, Cmu+0*Rtrue, '--k', linewidth=1.5)
	pl.plot(E+0*Ptrue, Rtrue, '--k', linewidth=1.5)
	pl.ylim([0,1.05*Rmax])
	pl.ylabel('$I[\mathcal{R};\overleftarrow{X}]$')
	pl.xlim([0,1.05*Pmax])
	pl.xlabel('$I[\overleftarrow{X};\overrightarrow{X}|\mathcal{R}]$')
	xlocs = pl.xticks()[0]; xlocs = xlocs[np.abs(xlocs-E)>0.1]
	xlabels = np.append(['$'+str(item)+'$' for item in xlocs],'$\mathbf{E}$')
	xlocs = np.append(xlocs,E)
	pl.xticks(xlocs,xlabels)
	ylocs = pl.yticks()[0]; ylocs = ylocs[np.abs(ylocs-Cmu)>0.1]
	ylabels = np.append(['$'+str(item)+'$' for item in ylocs],'$C_{\mu}$')
	ylocs = np.append(ylocs,Cmu)
	pl.yticks(ylocs,ylabels)
	pl.savefig(eM_name+time_dir+str(params)+'_PRD.pdf',bbox_inches='tight')
	#pl.show()
	pl.clf()

	# PIB figure
	#fig = pl.figure()
	pl.plot(btrue, Rtrue, '-o', linewidth=2)
	pl.plot(b2, R2, '--x', linewidth=2)
	pl.plot(b3, R3, '--x', linewidth=2)
	pl.plot(b4, R4, '--x', linewidth=2)
	pl.plot(b5, R5, '--x', linewidth=2)
	pl.plot(btrue, Cmu+0*Rtrue, '--k', linewidth=1.5)
	pl.xlim([0,betamax])
	pl.xlabel(r'$\beta$')
	pl.ylim([0,1.05*Rmax])
	pl.ylabel('$I[\mathcal{R};\overleftarrow{X}]$')
	ylocs = pl.yticks()[0]; ylocs = ylocs[np.abs(ylocs-Cmu)>0.1]
	ylabels = np.append(['$'+str(item)+'$' for item in ylocs],'$C_{\mu}$')
	ylocs = np.append(ylocs,Cmu)
	pl.yticks(ylocs,ylabels)
	pl.savefig(eM_name+time_dir+str(params)+'_PIB.pdf',bbox_inches='tight')
	#pl.show()
	pl.clf()

	return btrue, Rtrue, Ptrue, R2, P2, R3, P3, R4, P4, R5, P5

def makeFigs_lowsigmamu(eM_name,time_dir,params,betamax,numsteps):
	# ########
	btrue, Rtrue, Ptrue = CN(eM_name,time_dir,params,0.1,betamax,numsteps)
	b1, R1, P1 = OCF(eM_name,time_dir,params,1,1,0.1,betamax,numsteps)

	Rmax = np.max([np.max(R1),np.max(Rtrue)])
	Cmu = 2.71#np.max(Rtrue)
	Pmax = np.max(Ptrue)
	E = Pmax
	betamax = np.max(btrue)

	# PIC figure
	fig = pl.figure()
	pl.plot(Rtrue, Ptrue, '-o', linewidth=2)
	pl.plot(R1, P1, '--x', linewidth=2)
	pl.xlim([0,1.05*Rmax])
	pl.xlabel('$I[\overleftarrow{X};\mathcal{R}]$')
	pl.ylim([0,1.05*Pmax])
	pl.ylabel('$I[\mathcal{R};\overrightarrow{X}]$')
	xlocs = pl.xticks()[0]
	xlabels = np.append(['$'+str(item)+'$' for item in xlocs],'$C_{\mu}$')
	xlocs = np.append(xlocs,Cmu)
	pl.xticks(xlocs,xlabels)
	ylocs = pl.yticks()[0]
	ylabels = np.append(['$'+str(item)+'$' for item in ylocs],'$\mathbf{E}$')
	ylocs = np.append(ylocs,E)
	pl.yticks(ylocs,ylabels)
	pl.savefig(eM_name+time_dir+str(params)+'_PIC.pdf',bbox_inches='tight')
	#pl.show()
	pl.clf()

	# PRD figure
	fig = pl.figure()
	pl.plot(Pmax-Ptrue, Rtrue, '-o', linewidth=2)
	pl.plot(Pmax-P1, R1, '--x', linewidth=2)
	pl.ylim([0,1.05*Rmax])
	pl.plot(Ptrue,Cmu+0*Rtrue,'--k',linewidth=1.5)
	pl.ylabel('$I[\overleftarrow{X};\mathcal{R}]$')
	pl.xlim([0,1.05*Pmax])
	pl.plot(E+0*Ptrue,Rtrue,'--k',linewidth=1.5)
	pl.xlabel('$I[\overleftarrow{X};\overrightarrow{X}|\mathcal{R}]$')
	xlocs = pl.xticks()[0]; xlocs = xlocs[np.abs(xlocs-E)>0.1]
	xlabels = np.append(['$'+str(item)+'$' for item in xlocs],'$\mathbf{E}$')
	xlocs = np.append(xlocs,E)
	pl.xticks(xlocs,xlabels)
	ylocs = pl.yticks()[0]; ylocs = ylocs[np.abs(ylocs-Cmu)>0.1]
	ylabels = np.append(['$'+str(item)+'$' for item in ylocs],'$C_{\mu}$')
	ylocs = np.append(ylocs,Cmu)
	pl.yticks(ylocs,ylabels)
	pl.savefig(eM_name+time_dir+str(params)+'_PRD.pdf',bbox_inches='tight')
	#pl.show()
	pl.clf()

	# PIB figure
	fig = pl.figure()
	pl.semilogx(btrue, Rtrue, '-o', linewidth=2, label='CN')
	pl.semilogx(b1, R1, '--x', linewidth=2, label='OCF, L=1')
	pl.semilogx(btrue,Cmu+0*Rtrue,'--k',linewidth=1.5)
	pl.xlim([0,betamax])
	pl.xlabel(r'$\beta$')
	pl.ylim([0,1.05*Rmax])
	pl.ylabel('$I[\overleftarrow{X};\mathcal{R}]$')
	ylocs = pl.yticks()[0]; ylocs = ylocs[np.abs(ylocs-Cmu)>0.1]
	ylabels = np.append(['$'+str(item)+'$' for item in ylocs],'$C_{\mu}$')
	ylocs = np.append(ylocs,Cmu)
	pl.yticks(ylocs,ylabels)
	pl.savefig(eM_name+time_dir+str(params)+'_PIB.pdf',bbox_inches='tight')
	#pl.show()
	pl.clf()

	return btrue, Rtrue, Ptrue, R1, P1

##############################

# btrue, Rtrue, Ptrue, R2, P2, R3, P3, R4, P4, R5, P5 = makeFigs('Even','F',0.5,15,100)
# np.savez('Even0.5_PIB_v3.npz',btrue=btrue, Rtrue=Rtrue, Ptrue=Ptrue, R2=R2, P2=P2, R3=R3, P3=P3, R4=R4, P4=P4, R5=R5, P5=P5)

# btrue, Rtrue, Ptrue, R2, P2, R3, P3, R4, P4, R5, P5 = makeFigs('RIP','F',[0.5,0.5],15,100)
# np.savez('RIPF,p=q=0.5_PIB_v3.npz',btrue=btrue, Rtrue=Rtrue, Ptrue=Ptrue, R2=R2, P2=P2, R3=R3, P3=P3, R4=R4, P4=P4, R5=R5, P5=P5)

# btrue, Rtrue, Ptrue, R2, P2, R3, P3, R4, P4, R5, P5 = makeFigs('RIP','F',[0.5,0.2],15,200)
# np.savez('RIP,p=0.5,q=0.2_PIB_v2.npz',btrue=btrue, Rtrue=Rtrue, Ptrue=Ptrue, R2=R2, P2=P2, R3=R3, P3=P3, R4=R4, P4=P4, R5=R5, P5=P5)

# btrue, Rtrue, Ptrue, R2, P2, R3, P3, R4, P4, R5, P5 = makeFigs('RIP','R',[0.5,0.5],15,100)
# np.savez('RIPR,p=q=0.5_PIB_v2.npz',btrue=btrue, Rtrue=Rtrue, Ptrue=Ptrue, R2=R2, P2=P2, R3=R3, P3=P3, R4=R4, P4=P4, R5=R5, P5=P5)

# btrue, Rtrue, Ptrue, R2, P2, R3, P3, R4, P4, R5, P5 = makeFigs('RIP','R',[0.5,0.2],15,200)
# np.savez('RIP_R,p=0.5,q=0.2_PIB.npz',btrue=btrue, Rtrue=Rtrue, Ptrue=Ptrue, R2=R2, P2=P2, R3=R3, P3=P3, R4=R4, P4=P4, R5=R5, P5=P5)

# btrue, Rtrue, Ptrue, R2, P2, R3, P3, R4, P4, R5, P5 = makeFigs('Butterfly','F',15,20,100)
# np.savez('Butterfly_PIB.npz',btrue=btrue, Rtrue=Rtrue, Ptrue=Ptrue, R2=R2, P2=P2, R3=R3, P3=P3, R4=R4, P4=P4, R5=R5, P5=P5)

# btrue, Rtrue, Ptrue, R2, P2, R3, P3, R4, P4, R5, P5 = makeFigs('Butterfly','R',16,1000,500)
# np.savez('Butterfly_R_PIB2.npz',btrue=btrue, Rtrue=Rtrue, Ptrue=Ptrue, R2=R2, P2=P2, R3=R3, P3=P3, R4=R4, P4=P4, R5=R5, P5=P5)

# btrue, Rtrue, Ptrue, R1, P1 = makeFigs_lowsigmamu('GoldenMean','F',.5,15,100)
# np.savez('GoldenMean,p=0.5_PIB.npz',btrue=btrue, Rtrue=Rtrue, Ptrue=Ptrue, R1=R1, P1=P1)

#b1, R1, P1 = OCF(eM_name,time_dir,params,1,1,0.1,betamax,numsteps)
#btrue, Rtrue, Ptrue, R1, P1 = makeFigs_lowsigmamu('SNS','F',[.5,.5,20],100000,1000)
#np.savez('SNS_L=10.npz',b1=b1,R1=R1,P1=P1,btrue=btrue,Rtrue=Rtrue,Ptrue=Ptrue)

# btrue, Rtrue, Ptrue, R2, P2, R3, P3, R4, P4, R5, P5 = makeFigs('TentMap','F',[],15,100)
# np.savez('TentMap_PIB_v2.npz',btrue=btrue, Rtrue=Rtrue, Ptrue=Ptrue, R2=R2, P2=P2, R3=R3, P3=P3, R4=R4, P4=P4, R5=R5, P5=P5)
