from dataclasses import dataclass
import numpy as np

@dataclass
class Atom:
    element: str
    frac: np.ndarray

@dataclass
class Structure:
    cell: np.ndarray
    atoms: list
    name: str = "Structure"

    @property
    def volume(self):
        return float(abs(np.linalg.det(self.cell)))

    @property
    def reciprocal_cell(self):
        return 2*np.pi*np.linalg.inv(self.cell).T

    def cartesian_positions(self):
        f = np.array([a.frac for a in self.atoms], dtype=float)
        return f @ self.cell

    def expand(self, nx=1, ny=1, nz=1):
        new=[]
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    shift=np.array([ix,iy,iz],float)
                    for a in self.atoms:
                        new.append(Atom(a.element, (a.frac+shift)/[nx,ny,nz]))
        newcell=np.diag([nx,ny,nz]) @ self.cell
        return Structure(newcell,new,self.name+f" {nx}x{ny}x{nz}")

def lattice_parameters(cell):
    a,b,c=cell
    return (np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(c),
            np.degrees(np.arccos(np.dot(b,c)/(np.linalg.norm(b)*np.linalg.norm(c)))),
            np.degrees(np.arccos(np.dot(a,c)/(np.linalg.norm(a)*np.linalg.norm(c)))),
            np.degrees(np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))))
