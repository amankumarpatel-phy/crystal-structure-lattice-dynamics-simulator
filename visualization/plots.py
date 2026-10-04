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


def fft_crystal_projection(structure, grid_size=256, sigma_pixels=1.0):
    """Build a 2D x-y atomic-density projection and return its FFT power spectrum.

    The FFT is a mathematical Fourier-space visualization of the projected
    atomic density; it is not a replacement for the powder-XRD scattering model.
    """
    pos=structure.cartesian_positions()
    xy=pos[:,:2]
    xmin,xmax=xy[:,0].min(),xy[:,0].max()
    ymin,ymax=xy[:,1].min(),xy[:,1].max()
    dx=max((xmax-xmin)/max(grid_size-1,1),1e-6)
    dy=max((ymax-ymin)/max(grid_size-1,1),1e-6)
    density=np.zeros((grid_size,grid_size),dtype=float)
    ix=np.clip(((xy[:,0]-xmin)/(xmax-xmin+1e-12)*(grid_size-1)).astype(int),0,grid_size-1)
    iy=np.clip(((xy[:,1]-ymin)/(ymax-ymin+1e-12)*(grid_size-1)).astype(int),0,grid_size-1)
    density[iy,ix]+=1.0
    if sigma_pixels>0:
        from scipy.ndimage import gaussian_filter
        density=gaussian_filter(density,sigma=sigma_pixels)
    F=np.fft.fftshift(np.fft.fft2(density))
    power=np.abs(F)**2
    power/=power.max() if power.max()>0 else 1.0
    fx=np.fft.fftshift(np.fft.fftfreq(grid_size,d=dx))
    fy=np.fft.fftshift(np.fft.fftfreq(grid_size,d=dy))
    qx=2*np.pi*fx
    qy=2*np.pi*fy
    return density,power,qx,qy


def plot_fft_crystal_projection(structure, ax=None, grid_size=256, sigma_pixels=1.0):
    """Plot log-scaled 2D FFT intensity of the projected crystal density."""
    density,power,qx,qy=fft_crystal_projection(structure,grid_size,sigma_pixels)
    if ax is None:
        ax=plt.subplots()[1]
    extent=[qx.min(),qx.max(),qy.min(),qy.max()]
    ax.imshow(np.log1p(100*power),origin='lower',extent=extent,aspect='equal')
    ax.set_xlabel(r'$q_x$ (Å$^{-1}$)'); ax.set_ylabel(r'$q_y$ (Å$^{-1}$)')
    ax.set_title('2D FFT of Projected Atomic Density')
    return ax

def fft_ift_reconstruction(structure, grid_size=256, sigma_pixels=1.0, cutoff_Ainv=None):
    """Return real-space density, FFT, filtered FFT, IFT reconstruction and q axes."""
    density,power,qx,qy=fft_crystal_projection(structure,grid_size,sigma_pixels)
    F=np.fft.fftshift(np.fft.fft2(density))
    F_filtered=F.copy()
    if cutoff_Ainv is not None and cutoff_Ainv>0:
        QX,QY=np.meshgrid(qx,qy)
        mask=np.sqrt(QX**2+QY**2)<=cutoff_Ainv
        F_filtered*=mask
    reconstruction=np.fft.ifft2(np.fft.ifftshift(F_filtered)).real
    return density,F,F_filtered,reconstruction,qx,qy

def plot_ift_reconstruction(structure, ax=None, grid_size=256, sigma_pixels=1.0, cutoff_Ainv=None):
    """Plot the real-space image reconstructed from a filtered Fourier spectrum."""
    density,F,F_filtered,reconstruction,qx,qy=fft_ift_reconstruction(
        structure,grid_size,sigma_pixels,cutoff_Ainv
    )
    if ax is None:
        ax=plt.subplots()[1]
    pos=np.array(structure.cartesian_positions())
    xmin,xmax=pos[:,0].min(),pos[:,0].max()
    ymin,ymax=pos[:,1].min(),pos[:,1].max()
    ax.imshow(reconstruction,origin='lower',extent=[xmin,xmax,ymin,ymax],aspect='auto')
    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
    ax.set_title('IFT Reconstruction in Real Space')
    return ax

def radial_fft_profile(structure, grid_size=256, sigma_pixels=1.0):
    """Azimuthally average FFT power into a 1D q profile."""
    density,power,qx,qy=fft_crystal_projection(structure,grid_size,sigma_pixels)
    QX,QY=np.meshgrid(qx,qy)
    q=np.sqrt(QX**2+QY**2).ravel()
    p=power.ravel()
    bins=np.linspace(0,q.max(),min(180,grid_size//2))
    idx=np.digitize(q,bins)
    centers=0.5*(bins[:-1]+bins[1:])
    prof=np.array([p[idx==i].mean() if np.any(idx==i) else np.nan for i in range(1,len(bins))])
    return centers,prof

def plot_phonon_dispersion(ax=None, spring_constant=1.0, mass=1.0, points=400):
    """Monoatomic 1D nearest-neighbour harmonic-chain dispersion, in normalized units."""
    if ax is None:
        ax=plt.subplots()[1]
    q=np.linspace(-np.pi,np.pi,points)
    omega=2*np.sqrt(max(spring_constant,1e-12)/max(mass,1e-12))*np.abs(np.sin(q/2))
    ax.plot(q,omega)
    ax.set_xlabel(r'Wave vector $qa$')
    ax.set_ylabel(r'Angular frequency $\omega$ (normalized)')
    ax.set_title('1D Monoatomic Chain — Acoustic Phonon Dispersion')
    ax.grid(alpha=.2)
    return ax
