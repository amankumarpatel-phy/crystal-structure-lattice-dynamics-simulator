"""XRD geometry, peak analysis, size/strain and instrumental broadening."""
import numpy as np
from scipy.signal import find_peaks

def scherrer_size(beta_2theta_deg, theta_deg, wavelength_A=1.5406, K=0.9):
    beta=np.deg2rad(beta_2theta_deg); theta=np.deg2rad(theta_deg)
    if np.any(beta<=0): raise ValueError("Corrected FWHM must be positive.")
    return K*wavelength_A/(beta*np.cos(theta))

def correct_instrumental_broadening(beta_obs_deg,beta_inst_deg):
    obs=np.asarray(beta_obs_deg,float); inst=np.asarray(beta_inst_deg,float)
    if np.any(obs<=0) or np.any(inst<0): raise ValueError("FWHM values must be positive/non-negative.")
    return np.sqrt(np.maximum(obs**2-inst**2,0.0))

def caglioti_fwhm(two_theta_deg,U=0.0,V=0.0,W=0.0):
    tt=np.asarray(two_theta_deg,float); theta=np.deg2rad(tt/2.0)
    h2=U*np.tan(theta)**2+V*np.tan(theta)+W
    return np.sqrt(np.maximum(h2,0.0))

def bragg_two_theta(d_A,wavelength_A=1.5406,order=1):
    d=np.asarray(d_A,float); arg=order*wavelength_A/(2*d)
    if np.any(arg>1): raise ValueError("Bragg condition cannot be satisfied.")
    return np.rad2deg(2*np.arcsin(arg))

def cubic_d_spacing(a_A,h,k,l):
    n=h*h+k*k+l*l
    if n<=0: raise ValueError("(h,k,l) cannot all be zero.")
    return float(a_A)/np.sqrt(n)

def lattice_parameter_from_peak(two_theta_deg,h,k,l,wavelength_A=1.5406,order=1):
    theta=np.deg2rad(np.asarray(two_theta_deg,float)/2); n=h*h+k*k+l*l
    return wavelength_A*np.sqrt(n)/(2*np.sin(theta)*order)

def detect_peaks(two_theta_deg,intensity,prominence=0.05,distance=5):
    x=np.asarray(two_theta_deg,float); y=np.asarray(intensity,float)
    if x.size!=y.size or x.size<3: raise ValueError("2θ and intensity must have same length and at least 3 points.")
    yn=(y-y.min())/(y.max()-y.min()+1e-15); idx,_=find_peaks(yn,prominence=prominence,distance=distance)
    return x[idx],y[idx],idx

def williamson_hall(two_theta_deg,fwhm_deg,wavelength_A=1.5406,K=0.9):
    tt=np.asarray(two_theta_deg,float); beta=np.deg2rad(np.asarray(fwhm_deg,float)); theta=np.deg2rad(tt/2)
    x=4*np.sin(theta); y=beta*np.cos(theta)
    if len(x)<2: raise ValueError("At least two peaks are required.")
    slope,intercept=np.polyfit(x,y,1); size_A=K*wavelength_A/intercept if intercept>0 else np.inf
    return x,y,slope,intercept,size_A

def r_factor(y_obs,y_calc):
    yo=np.asarray(y_obs,float); yc=np.asarray(y_calc,float); return 100*np.sum(np.abs(yo-yc))/(np.sum(np.abs(yo))+1e-15)
