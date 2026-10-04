import numpy as np
import matplotlib.pyplot as plt

def plot_structure(structure,ax=None):
    p=structure.cartesian_positions(); ax=ax or plt.figure().add_subplot(111,projection='3d')
    ax.scatter(p[:,0],p[:,1],p[:,2],s=70)
    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)'); ax.set_zlabel('z (Å)'); ax.set_title(structure.name)
    return ax

def plot_xrd(tt,I,ax=None,title='Simulated Powder XRD'):
    ax=ax or plt.subplots()[1]; ax.plot(tt,I); ax.set_xlabel(r'2θ (degrees)'); ax.set_ylabel('Normalized intensity'); ax.set_title(title); ax.grid(alpha=.2); return ax

def reciprocal_lattice_points(structure, hmax=4):
    """Return reciprocal-lattice points G = h b1 + k b2 + l b3."""
    b=structure.reciprocal_cell
    pts=[]; labels=[]
    for h in range(-hmax,hmax+1):
        for k in range(-hmax,hmax+1):
            for l in range(-hmax,hmax+1):
                pts.append(h*b[0]+k*b[1]+l*b[2])
                labels.append((h,k,l))
    return np.asarray(pts,float), labels

def plot_reciprocal_lattice(structure, ax=None, hmax=4, plane_hkl=(1,0,0), plane_extent=8.0):
    """Visualize reciprocal-lattice points and a selected (hkl) reciprocal-space plane."""
    if ax is None:
        ax=plt.figure().add_subplot(111,projection='3d')
    pts,labels=reciprocal_lattice_points(structure,hmax)
    ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=12,alpha=0.45,label='Reciprocal lattice points')
    h,k,l=plane_hkl
    G=h*structure.reciprocal_cell[0]+k*structure.reciprocal_cell[1]+l*structure.reciprocal_cell[2]
    n=np.linalg.norm(G)
    if n>0:
        nh=G/n
        ref=np.array([1.,0.,0.]) if abs(nh[0])<0.9 else np.array([0.,1.,0.])
        u=np.cross(nh,ref); u/=np.linalg.norm(u); v=np.cross(nh,u)
        s=np.linspace(-plane_extent,plane_extent,2)
        U,V=np.meshgrid(s,s)
        # Plane G·r = |G|²/2, midway between origin and the first reciprocal point G.
        center=G/2
        P=center[None,None,:]+U[:,:,None]*u[None,None,:]+V[:,:,None]*v[None,None,:]
        ax.plot_surface(P[:,:,0],P[:,:,1],P[:,:,2],alpha=0.18,label=f'Reciprocal plane ({h}{k}{l})')
    ax.set_xlabel(r'$G_x$ (Å$^{-1}$)'); ax.set_ylabel(r'$G_y$ (Å$^{-1}$)'); ax.set_zlabel(r'$G_z$ (Å$^{-1}$)')
    ax.set_title(f'Reciprocal Lattice & ({h}{k}{l}) Plane')
    return ax
