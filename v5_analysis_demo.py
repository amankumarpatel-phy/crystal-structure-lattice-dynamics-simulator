import numpy as np
from analysis.instrument import InstrumentConfig
from analysis.xrd import correct_instrumental_broadening,scherrer_size,williamson_hall
from analysis.refinement import fit_single_peak,pseudo_voigt
from analysis.indexing import index_cubic_peaks

rng=np.random.default_rng(7)
x=np.linspace(20,90,3501)
true_centers=[28.44,47.30,56.12,69.13]
y=np.full_like(x,100.0)
for c,a,f,e in zip(true_centers,[1000,650,500,360],[0.20,0.24,0.28,0.34],[0.35,0.35,0.40,0.45]):
    y+=pseudo_voigt(x,a,c,f,e)
y+=rng.normal(0,10,x.size)
mask=(x>=27.8)&(x<=29.1)
fit=fit_single_peak(x[mask],y[mask],28.44,model='pseudo_voigt')
print('Representative pseudo-Voigt fit:',fit)
inst=InstrumentConfig(wavelength_A=1.5406,W=0.0004)
tt=np.array(true_centers); obs=np.array([0.20,0.24,0.28,0.34])
sample=correct_instrumental_broadening(obs,inst.instrumental_fwhm(tt))
print('Instrument-corrected FWHM:',sample)
print('Scherrer sizes (Å):',scherrer_size(sample,tt/2))
wh=williamson_hall(tt,sample)
print('Williamson-Hall size (Å):',wh[4],'microstrain:',wh[2])
print('FCC indexing example:')
for row in index_cubic_peaks(tt,5.430,centering='fcc'): print(row)
