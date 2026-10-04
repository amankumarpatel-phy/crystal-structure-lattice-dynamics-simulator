"""Robust CSV/XY import for two-column powder XRD data."""
import numpy as np
import pandas as pd

def load_xrd_csv(source):
    if hasattr(source,'read'):
        df=pd.read_csv(source,comment='#',sep=None,engine='python')
    else:
        try: df=pd.read_csv(source,comment='#',sep=None,engine='python')
        except Exception: df=pd.read_csv(source,comment='#',header=None,names=['2theta','intensity'])
    if df.shape[1]<2: raise ValueError('XRD file must contain at least two columns: 2θ and intensity.')
    x=pd.to_numeric(df.iloc[:,0],errors='coerce').to_numpy(); y=pd.to_numeric(df.iloc[:,1],errors='coerce').to_numpy()
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    if x.size<5: raise ValueError('Not enough numeric XRD points found.')
    order=np.argsort(x)
    return x[order],y[order],df.iloc[:,0].name,df.iloc[:,1].name