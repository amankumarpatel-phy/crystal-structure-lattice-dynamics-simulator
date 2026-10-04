import numpy as np
from .structure import Structure

FORM_FACTORS={
'O':([3.0485,2.2868,1.5463,0.867],[13.2771,5.7011,0.3239,32.9089],0.2508),
'C':([2.31,1.02,1.5886,0.865],[20.8439,10.2075,0.5687,51.6512],0.2156),
'Si':([6.2915,3.0353,1.9891,1.541],[2.4386,32.3337,0.6785,81.6937],1.1407),
'Al':([4.17448,3.3876,1.20296,0.528137],[1.93816,4.14553,0.228753,8.28524],0.706786),
'Fe':([11.7695,7.3573,3.5222,2.3045],[4.7611,0.3072,15.3535,76.8805],1.0369),
'Cu':([13.338,7.1676,5.6158,1.6735],[3.5828,0.247,11.3966,64.8126],1.191),
'Ni':([12.8376,7.292,5.6118,2.1135],[3.8785,0.2565,11.2004,71.4],0.600),
'Na':([3.2565,3.9362,1.3998,1.0032],[2.6671,6.1153,0.2001,14.039],0.4040),
'Cl':([11.4604,7.1964,6.2556,1.6455],[0.0104,1.1662,18.5194,47.7784],-9.5574),
}

def atomic_form_factor(element, s):
    if element not in FORM_FACTORS:
        return np.full_like(np.asarray(s,float), {'H':1.,'He':2.}.get(element,0.), dtype=float)
    a,b,c=FORM_FACTORS[element]; s=np.asarray(s,float)
    return sum(ai*np.exp(-bi*s*s) for ai,bi in zip(a,b))+c

def structure_factor(structure, hkl, wavelength=1.5406, two_theta_deg=30, B=0.0):
    theta=np.radians(two_theta_deg/2); s=np.sin(theta)/wavelength
    q=np.dot(np.asarray(hkl,float),structure.reciprocal_cell)
    amp=0j
    for atom in structure.atoms:
        f=atomic_form_factor(atom.element,np.array([s]))[0]
        dw=np.exp(-B*s*s) if B else 1.0
        amp += f*dw*np.exp(1j*np.dot(q,atom.frac @ structure.cell))
    return amp

def d_spacing(structure,hkl):
    g=np.dot(np.asarray(hkl,float),structure.reciprocal_cell)
    return 2*np.pi/np.linalg.norm(g)

def bragg_two_theta(d,wavelength=1.5406,order=1):
    x=order*wavelength/(2*d)
    if x>1: return np.nan
    return np.degrees(2*np.arcsin(x))

def powder_pattern(structure,wavelength=1.5406,two_theta=np.linspace(10,100,3601),hmax=6,B=0.0,peak_width=0.08):
    y=np.zeros_like(two_theta,dtype=float); peaks=[]
    for h in range(0,hmax+1):
      for k in range(-hmax,hmax+1):
       for l in range(-hmax,hmax+1):
        if (h,k,l)==(0,0,0): continue
        d=d_spacing(structure,(h,k,l)); tt=bragg_two_theta(d,wavelength)
        if not np.isfinite(tt) or tt<two_theta.min()-1 or tt>two_theta.max()+1: continue
        F=structure_factor(structure,(h,k,l),wavelength,tt,B); I=abs(F)**2
        if I<1e-6: continue
        y += I*np.exp(-0.5*((two_theta-tt)/peak_width)**2)
        peaks.append((tt,I,(h,k,l),d))
    if y.max()>0: y/=y.max()
    return two_theta,y,sorted(peaks,key=lambda x:x[0])
