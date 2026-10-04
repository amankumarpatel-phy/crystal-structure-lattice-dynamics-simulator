"""Small dependency-free CIF reader for common single-block crystallographic files."""
import shlex, re
import numpy as np
from .structure import Structure, Atom

def _num(s):
    s=str(s).strip().strip("'\"")
    s=re.sub(r"\([^)]*\)$","",s)
    return float(s)

def read_cif(path):
    with open(path,'r',encoding='utf-8',errors='ignore') as f: lines=f.readlines()
    tags={}; loops=[]; i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line or line.startswith('#'): i+=1; continue
        if line.lower().startswith('loop_'):
            i+=1; headers=[]
            while i<len(lines) and lines[i].strip().startswith('_'):
                headers.append(lines[i].strip().split()[0]); i+=1
            rows=[]; buf=[]
            while i<len(lines):
                x=lines[i].strip()
                if not x or x.startswith('#'): i+=1; continue
                if x.lower()=='loop_' or x.startswith('_'): break
                vals=shlex.split(x, comments=True)
                buf.extend(vals)
                while len(buf)>=len(headers):
                    rows.append(buf[:len(headers)]); buf=buf[len(headers):]
                i+=1
            loops.append((headers,rows)); continue
        if line.startswith('_'):
            vals=shlex.split(line, comments=True)
            if len(vals)>=2: tags[vals[0]]=vals[1]
        i+=1
    a,b,c=[_num(tags[f'_cell_length_{x}']) for x in 'abc']
    al,be,ga=[np.radians(_num(tags[f'_cell_angle_{x}'])) for x in ('alpha','beta','gamma')]
    va=np.array([a,0,0.]); vb=np.array([b*np.cos(ga),b*np.sin(ga),0.])
    cx=c*np.cos(be); cy=c*(np.cos(al)-np.cos(be)*np.cos(ga))/np.sin(ga)
    cz=np.sqrt(max(c*c-cx*cx-cy*cy,0)); cell=np.array([va,vb,[cx,cy,cz]])
    atoms=[]
    headers=rows=None
    for h,r in loops:
        if '_atom_site_fract_x' in h and '_atom_site_fract_y' in h and '_atom_site_fract_z' in h:
            headers,rows=h,r; break
    if rows is None: raise ValueError('No atom-site fractional coordinates found in CIF')
    ix={h:j for j,h in enumerate(headers)}
    for r in rows:
        sym=None
        for k in ('_atom_site_type_symbol','_atom_site_label'):
            if k in ix: sym=re.sub(r'[^A-Za-z]','',r[ix[k]]); break
        if not sym: continue
        sym=sym[0].upper()+sym[1:].lower()
        atoms.append(Atom(sym,np.array([_num(r[ix['_atom_site_fract_x']]),_num(r[ix['_atom_site_fract_y']]),_num(r[ix['_atom_site_fract_z']])],float)))
    return Structure(cell,atoms,tags.get('_chemical_name_common', 'CIF Structure').strip("'\""))
