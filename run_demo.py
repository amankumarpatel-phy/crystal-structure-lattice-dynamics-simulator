from crystal.cif import read_cif
from crystal.scattering import powder_pattern
s=read_cif('data/Si.cif')
tt,I,peaks=powder_pattern(s)
print('Structure:',s.name)
print('Atoms:',len(s.atoms))
print('Top simulated reflections:')
for p in sorted(peaks,key=lambda z:z[1],reverse=True)[:10]: print(f'2θ={p[0]:.3f}°, I={p[1]:.2f}, hkl={p[2]}, d={p[3]:.4f} Å')