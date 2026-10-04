# Crystal Structure & Lattice Dynamics Simulator — V5

**Research-oriented computational crystallography workstation in Python**

## Project scope
This release extends the earlier real-space/reciprocal-space simulator into a reproducible XRD analysis workflow:

**CIF → 3D crystal → structure factor → powder XRD → experimental data → peak fitting → indexing → instrumental correction → size/strain analysis**

The software deliberately separates transparent physics calculations from empirical profile fitting. It is intended for education, portfolio work, method development, and exploratory analysis; it is **not** a replacement for a validated Rietveld package.

## V5 additions

- CIF input and 3D unit-cell visualization
- Atomic form factors and explicit structure factors
- Simulated powder XRD and Debye–Waller attenuation
- Robust two-column CSV/XY import
- Experimental peak detection
- Gaussian and pseudo-Voigt peak fitting
- Caglioti instrumental broadening model
- Zero-shift correction metadata
- Scherrer crystallite-size estimation after broadening correction
- Williamson–Hall size/strain analysis
- Cubic peak indexing with primitive/BCC/FCC selection rules
- Lattice parameter estimation from indexed peaks
- Lightweight whole-pattern cubic lattice-parameter refinement hook
- RMSE and Rwp fit-quality metrics
- Research-oriented GUI in Streamlit
- Automated validation tests

## Physics used

### Structure factor
F(hkl) = Σ_j f_j exp[2πi(hx_j + ky_j + lz_j)]

### Bragg condition
2 d_hkl sin(theta) = lambda

### Scherrer estimate
D = K lambda / (beta cos(theta))

where beta is the sample broadening in radians of 2theta after instrumental correction.

### Williamson–Hall
beta cos(theta) = K lambda / D + 4 epsilon sin(theta)

### Caglioti instrumental width
H² = U tan²(theta) + V tan(theta) + W

## Install
~~~bash
pip install -r requirements.txt
~~~

## Run GUI
~~~bash
streamlit run app.py
~~~

## Run analysis demo
~~~bash
python v5_analysis_demo.py
~~~

## Run tests
~~~bash
python -m pytest -q
~~~

## Experimental CSV format
Two numeric columns are expected:

~~~text
2theta,intensity
20.00,120
20.02,123
...
~~~

Comment lines starting with # are ignored.

## Important scientific limitations

This package does **not** implement a full Rietveld refinement. It does not yet provide a validated treatment of all effects such as preferred orientation, absorption, axial divergence, Kα doublet splitting, detector geometry, anisotropic broadening, complex profile functions, or full background models. The V5 whole-pattern refinement is intentionally a lightweight demonstration hook.

## Suggested V6 research extensions

- CIF symmetry/space-group operations and systematic absences
- Kα1/Kα2 doublet model
- Thompson–Cox–Hastings pseudo-Voigt profile
- Zero/background/profile parameter refinement with uncertainties
- Pawley/Le Bail style whole-pattern fitting
- Anisotropic size/strain models
- Rietveld-compatible calculation layer
- Experimental metadata export and provenance report


---

<div align="center">

**© 2026 Aman Kumar Patel**  
*Made with ❤️ by Aman Kumar Patel*

</div>
