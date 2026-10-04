import numpy as np
from crystal.cif import read_cif
from crystal.scattering import structure_factor,powder_pattern
from analysis.xrd import bragg_two_theta,cubic_d_spacing,caglioti_fwhm,correct_instrumental_broadening,scherrer_size,williamson_hall
from analysis.refinement import pseudo_voigt,fit_single_peak
from analysis.indexing import index_cubic_peaks
from analysis.instrument import InstrumentConfig

def test_cif_and_volume():
    s=read_cif('data/Si.cif'); assert s.volume>150 and len(s.atoms)>=1

def test_structure_factor_zero_reflection():
    s=read_cif('data/Si.cif'); F=structure_factor(s,(0,0,0),two_theta_deg=10); assert abs(F)>0

def test_bragg_and_d_spacing():
    d=cubic_d_spacing(5.43,1,1,1); tt=bragg_two_theta(d); assert 28<tt<29

def test_caglioti_and_correction():
    inst=caglioti_fwhm(np.array([30.,60.]),0,0,0.01); assert np.allclose(inst,0.1)
    corrected=correct_instrumental_broadening(0.2,0.1); assert np.isclose(corrected,np.sqrt(0.03))

def test_scherrer_and_wh():
    size=scherrer_size(0.2,20); assert size>0
    x,y,slope,intercept,size_A=williamson_hall([30,50,70],[0.2,0.25,0.3]); assert len(x)==3 and np.isfinite(size_A)

def test_peak_fit():
    x=np.linspace(27.5,29.5,501); y=pseudo_voigt(x,100,28.44,0.20,0.4,5); fit=fit_single_peak(x,y,28.44)
    assert abs(fit['center']-28.44)<0.03 and abs(fit['fwhm']-0.20)<0.05 and fit['r2']>0.985

def test_fcc_indexing():
    rows=index_cubic_peaks([28.44,47.30,56.12],5.430,centering='fcc'); assert all(r['indexed'] for r in rows); assert rows[0]['hkl']==(1,1,1)

def test_powder_pattern():
    s=read_cif('data/Si.cif'); tt,I,peaks=powder_pattern(s,hmax=4); assert len(tt)==len(I) and I.max()>0 and len(peaks)>0

def test_instrument_config():
    inst=InstrumentConfig(W=0.01,zero_shift_deg=0.12); assert np.isclose(inst.instrumental_fwhm(30),0.1) and np.isclose(inst.corrected_two_theta(30),29.88)
