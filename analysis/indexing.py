"""Peak indexing helpers for cubic materials."""
import numpy as np
from .xrd import cubic_d_spacing, bragg_two_theta

ALLOWED_RULES={
    'primitive': lambda h,k,l: True,
    'bcc': lambda h,k,l: (h+k+l)%2==0,
    'fcc': lambda h,k,l: (h%2==k%2==l%2),
}

def index_cubic_peaks(two_theta_deg,a_A,wavelength_A=1.5406,max_index=8,tolerance_deg=0.25,centering='primitive'):
    obs=np.asarray(two_theta_deg,float); rule=ALLOWED_RULES.get(centering,ALLOWED_RULES['primitive']); candidates=[]
    for h in range(0,max_index+1):
      for k in range(0,max_index+1):
       for l in range(0,max_index+1):
        if h==k==l==0 or not rule(h,k,l): continue
        d=cubic_d_spacing(a_A,h,k,l); arg=wavelength_A/(2*d)
        if arg<=1.0:
            tt=bragg_two_theta(d,wavelength_A)
            if np.isfinite(tt): candidates.append((float(tt),(h,k,l),d))
    out=[]
    for value in obs:
        tt,hkl,d=min(candidates,key=lambda z:abs(z[0]-value))
        out.append({'observed_2theta':float(value),'calculated_2theta':tt,'hkl':hkl,'d_A':d,'delta_deg':abs(value-tt),'indexed':abs(value-tt)<=tolerance_deg})
    return out