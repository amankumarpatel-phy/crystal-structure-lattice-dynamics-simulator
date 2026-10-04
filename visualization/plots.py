import numpy as np
import matplotlib.pyplot as plt

def plot_structure(structure,ax=None):
    p=structure.cartesian_positions(); ax=ax or plt.figure().add_subplot(111,projection='3d')
    ax.scatter(p[:,0],p[:,1],p[:,2],s=70)
    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)'); ax.set_zlabel('z (Å)'); ax.set_title(structure.name)
    return ax

def plot_xrd(tt,I,ax=None,title='Simulated Powder XRD'):
    ax=ax or plt.subplots()[1]; ax.plot(tt,I); ax.set_xlabel(r'2θ (degrees)'); ax.set_ylabel('Normalized intensity'); ax.set_title(title); ax.grid(alpha=.2); return ax