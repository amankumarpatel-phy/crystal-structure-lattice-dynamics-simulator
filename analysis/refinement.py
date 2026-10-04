"""Transparent profile fitting and lightweight powder-pattern refinement hooks."""
import numpy as np
from scipy.optimize import least_squares

def gaussian(x,amp,center,sigma,background=0.0):
    return background+amp*np.exp(-0.5*((x-center)/sigma)**2)

def lorentzian(x,amp,center,gamma,background=0.0):
    return background+amp/(1.0+((x-center)/gamma)**2)

def pseudo_voigt(x,amp,center,fwhm,eta,background=0.0):
    f=max(float(fwhm),1e-9); eta=float(np.clip(eta,0,1))
    sigma=f/(2*np.sqrt(2*np.log(2))); gamma=f/2
    g=np.exp(-0.5*((x-center)/sigma)**2); l=1/(1+((x-center)/gamma)**2)
    return background+amp*((1-eta)*g+eta*l)

def fit_single_peak(x,y,center_guess=None,model='pseudo_voigt'):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if x.size<8 or x.size!=y.size: raise ValueError("Peak window is too small or malformed.")
    i=int(np.argmax(y)) if center_guess is None else int(np.argmin(abs(x-center_guess)))
    bg=float(np.percentile(y,10)); amp=max(float(y[i]-bg),1e-12)
    span=max(float(x[-1]-x[0]),1e-3); f0=max(span/8,3*(x[1]-x[0]))
    if model=='gaussian':
        def fun(p): return gaussian(x,p[0],p[1],p[2],p[3])-y
        p0=[amp,x[i],f0/(2*np.sqrt(2*np.log(2))),bg]; lo=[0,x.min(),1e-8,-np.inf]; hi=[np.inf,x.max(),np.inf,np.inf]
        res=least_squares(fun,p0,bounds=(lo,hi)); a,c,s,b=res.x; fwhm=2*np.sqrt(2*np.log(2))*s; eta=0.0
    else:
        def fun(p): return pseudo_voigt(x,p[0],p[1],p[2],p[4],p[3])-y
        p0=[amp,x[i],f0,0.5,bg]; lo=[0,x.min(),1e-8,0,-np.inf]; hi=[np.inf,x.max(),span,1,np.inf]
        res=least_squares(fun,p0,bounds=(lo,hi)); a,c,fwhm,eta,b=res.x
    ss_res=np.sum(res.fun**2); ss_tot=np.sum((y-y.mean())**2)+1e-15
    return {'amplitude':float(a),'center':float(c),'fwhm':float(fwhm),'eta':float(eta),'background':float(b),'r2':float(1-ss_res/ss_tot),'rmse':float(np.sqrt(np.mean(res.fun**2))),'success':bool(res.success)}

def bragg_for_a(a,h,k,l,wavelength_A=1.5406):
    n=h*h+k*k+l*l
    if n<=0:return np.nan
    d=a/np.sqrt(n); q=wavelength_A/(2*d)
    if q>1:return np.nan
    return np.degrees(2*np.arcsin(q))

def refine_cubic_lattice_parameter(two_theta,intensity,hkls,a0,wavelength_A=1.5406,peak_width=0.10):
    x=np.asarray(two_theta,float); y=np.asarray(intensity,float); yscale=max(y.max(),1e-12); yn=y/yscale
    p0=np.array([a0,1.0,0.02],float)
    def calc(p):
        a,scale,bg=p; out=np.full_like(x,bg,dtype=float)
        for h,k,l in hkls:
            tt=bragg_for_a(a,h,k,l,wavelength_A)
            if np.isfinite(tt): out+=scale*np.exp(-0.5*((x-tt)/peak_width)**2)
        return out
    res=least_squares(lambda p:calc(p)-yn,p0,bounds=([a0*0.8,0,-1],[a0*1.2,10,1]))
    return {'a_A':float(res.x[0]),'scale':float(res.x[1]),'background':float(res.x[2]),'rmse':float(np.sqrt(np.mean(res.fun**2))),'success':bool(res.success)}
