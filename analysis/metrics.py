import numpy as np

def rmse(y_obs,y_calc):
    a=np.asarray(y_obs,float); b=np.asarray(y_calc,float)
    return float(np.sqrt(np.mean((a-b)**2)))

def rwp(y_obs,y_calc,weights=None):
    y=np.asarray(y_obs,float); yc=np.asarray(y_calc,float)
    w=np.ones_like(y) if weights is None else np.asarray(weights,float)
    return float(100*np.sqrt(np.sum(w*(y-yc)**2)/(np.sum(w*y*y)+1e-15)))