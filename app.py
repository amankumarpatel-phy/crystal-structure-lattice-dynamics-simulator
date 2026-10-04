import os, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from crystal.cif import read_cif
from crystal.structure import Structure, Atom
from crystal.scattering import powder_pattern
from visualization.plots import (
    plot_structure, plot_xrd, plot_reciprocal_lattice,
    plot_fft_crystal_projection, plot_ift_reconstruction,
    radial_fft_profile, fft_ift_reconstruction,
    plot_phonon_dispersion, plot_fft_phase, plot_fft_autocorrelation,
    plot_diatomic_dispersion
)
from analysis.io import load_xrd_csv
from analysis.xrd import detect_peaks, correct_instrumental_broadening, scherrer_size, williamson_hall, cubic_d_spacing, bragg_two_theta
from analysis.refinement import fit_single_peak, pseudo_voigt
from analysis.instrument import InstrumentConfig
from analysis.indexing import index_cubic_peaks
from scipy.signal import savgol_filter
from scipy.stats import linregress

st.set_page_config(page_title='Crystal Structure & Lattice Dynamics Simulator', layout='wide')
st.title('Crystal Structure & Lattice Dynamics Simulator')
st.caption('CIF → Real Space → Reciprocal/Fourier Space → XRD → Peak Analysis → Lattice Dynamics')


def build_simulation_structure(base, supercell=1, defect_fraction=0.0, thermal_sigma=0.0, seed=42):
    """Create a reproducible visualization/simulation structure with optional disorder."""
    sim = base.expand(supercell, supercell, supercell) if supercell > 1 else base
    rng = np.random.default_rng(seed)
    atoms = list(sim.atoms)

    if defect_fraction > 0 and len(atoms) > 1:
        n_remove = min(len(atoms)-1, int(round(len(atoms) * defect_fraction / 100.0)))
        if n_remove:
            remove = set(rng.choice(len(atoms), size=n_remove, replace=False))
            atoms = [a for i, a in enumerate(atoms) if i not in remove]

    if thermal_sigma > 0 and atoms:
        inv_cell = np.linalg.inv(sim.cell)
        displaced = []
        for a in atoms:
            delta_cart = rng.normal(0.0, thermal_sigma, 3)
            delta_frac = delta_cart @ inv_cell
            displaced.append(Atom(a.element, (np.asarray(a.frac) + delta_frac) % 1.0))
        atoms = displaced

    return Structure(sim.cell.copy(), atoms, sim.name + ' | simulated disorder')


# ---------------- Sidebar: grouped controls ----------------
with st.sidebar:
    st.header('Simulation Controls')

    with st.expander('Crystal / Disorder', expanded=True):
        supercell = st.slider('Supercell replication', 1, 3, 1)
        defect_fraction = st.slider('Vacancy / defect fraction (%)', 0.0, 20.0, 0.0, 0.5)
        thermal_sigma = st.slider('Thermal displacement σ (Å)', 0.0, 0.25, 0.0, 0.01)

    with st.expander('Reciprocal Space', expanded=False):
        reciprocal_hmax = st.slider('Reciprocal lattice |h,k,l|', 1, 6, 4)
        plane_h = st.number_input('Plane h', value=1, min_value=-6, max_value=6, step=1)
        plane_k = st.number_input('Plane k', value=0, min_value=-6, max_value=6, step=1)
        plane_l = st.number_input('Plane l', value=0, min_value=-6, max_value=6, step=1)

    with st.expander('Fourier / FFT / IFT', expanded=False):
        fft_grid = st.select_slider('FFT grid size', [64, 128, 256, 512], value=256)
        fft_sigma = st.slider('Atomic-density smoothing (pixels)', 0.0, 3.0, 1.0, 0.25)
        ift_cutoff = st.slider('IFT low-pass cutoff |q| (Å⁻¹)', 0.1, 10.0, 3.0, 0.1)

    with st.expander('XRD Instrument', expanded=False):
        wavelength = st.number_input('X-ray wavelength λ (Å)', value=1.5406, min_value=0.1)
        hmax = st.slider('Maximum |h,k,l| for XRD', 1, 8, 5)
        B = st.number_input('Debye–Waller B (Å²)', value=0.0, min_value=0.0)
        width = st.number_input('Simulated Gaussian width (°)', value=0.10, min_value=0.01)
        U = st.number_input('Caglioti U', value=0.0, format='%.6f')
        V = st.number_input('Caglioti V', value=0.0, format='%.6f')
        W = st.number_input('Caglioti W (deg²)', value=0.01, min_value=0.0, format='%.6f')
        zero = st.number_input('2θ zero shift (°)', value=0.0, format='%.4f')

    cif = st.file_uploader('Upload CIF file', type=['cif'])

inst = InstrumentConfig(wavelength_A=wavelength, U=U, V=V, W=W, zero_shift_deg=zero)

if cif:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cif') as f:
        f.write(cif.getbuffer())
        path = f.name
    try:
        base_structure = read_cif(path)
    finally:
        os.unlink(path)
else:
    base_structure = read_cif('data/Si.cif')

sim_structure = build_simulation_structure(
    base_structure, supercell, defect_fraction, thermal_sigma
)

# ---------------- Main scientific sections ----------------
tab_structure, tab_fourier, tab_xrd, tab_exp, tab_size, tab_dyn = st.tabs([
    '1. Crystal Structure',
    '2. Reciprocal & Fourier Space',
    '3. Simulated XRD',
    '4. Experimental XRD',
    '5. Size / Strain',
    '6. Lattice Dynamics'
])

with tab_structure:
    st.subheader('Real-Space Crystal Structure')
    c1, c2 = st.columns([1.6, 1])
    with c1:
        fig = plt.figure(figsize=(7, 5))
        plot_structure(sim_structure, fig.add_subplot(111, projection='3d'))
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        a, b, c = [np.linalg.norm(v) for v in sim_structure.cell]
        st.metric('Atoms in model', len(sim_structure.atoms))
        st.metric('Unit-cell volume', f'{sim_structure.volume:.3f} Å³')
        st.write(f'**a:** {a:.4f} Å')
        st.write(f'**b:** {b:.4f} Å')
        st.write(f'**c:** {c:.4f} Å')
        st.write(f'**Supercell:** {supercell} × {supercell} × {supercell}')
        st.write(f'**Defects:** {defect_fraction:.1f}%')
        st.write(f'**Thermal σ:** {thermal_sigma:.3f} Å')
    st.info('The disorder controls modify the model used by the visualization and Fourier/XRD simulation. The original CIF structure remains unchanged.')
    st.subheader('Atomic Coordinates')
    atom_rows = [
        {'#': i+1, 'Element': a.element, 'fx': float(a.frac[0]), 'fy': float(a.frac[1]), 'fz': float(a.frac[2])}
        for i, a in enumerate(sim_structure.atoms)
    ]
    atom_df = pd.DataFrame(atom_rows)
    st.dataframe(atom_df, use_container_width=True, height=240)
    st.download_button(
        'Download atomic coordinates CSV',
        atom_df.to_csv(index=False).encode(),
        'atomic_coordinates.csv',
        'text/csv'
    )


with tab_fourier:
    st.subheader('Reciprocal Lattice')
    c1, c2 = st.columns(2)
    with c1:
        fig = plt.figure(figsize=(6, 5))
        plot_reciprocal_lattice(
            sim_structure,
            fig.add_subplot(111, projection='3d'),
            hmax=reciprocal_hmax,
            plane_hkl=(plane_h, plane_k, plane_l)
        )
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.markdown('**Reciprocal-space relation**')
        st.latex(r'\mathbf{G}=h\mathbf{b}_1+k\mathbf{b}_2+l\mathbf{b}_3')
        st.latex(r'\mathbf{G}_{hkl}\cdot\mathbf{r}=|\mathbf{G}_{hkl}|^2/2')
        st.write('The highlighted plane represents the reciprocal-space plane associated with the selected (hkl) reflection.')
        hkl_norm = plane_h**2 + plane_k**2 + plane_l**2
        if hkl_norm > 0:
            d_selected = 1.0 / np.sqrt(
                np.dot(
                    plane_h*sim_structure.reciprocal_cell[0] +
                    plane_k*sim_structure.reciprocal_cell[1] +
                    plane_l*sim_structure.reciprocal_cell[2],
                    plane_h*sim_structure.reciprocal_cell[0] +
                    plane_k*sim_structure.reciprocal_cell[1] +
                    plane_l*sim_structure.reciprocal_cell[2]
                )
            ) * 2*np.pi
            Gvec = (plane_h*sim_structure.reciprocal_cell[0] +
                    plane_k*sim_structure.reciprocal_cell[1] +
                    plane_l*sim_structure.reciprocal_cell[2])
            st.metric('Selected d-spacing', f'{d_selected:.4f} Å')
            st.metric('|G|', f'{np.linalg.norm(Gvec):.4f} Å⁻¹')
            try:
                st.metric('Bragg 2θ', f'{float(bragg_two_theta(d_selected, wavelength)):.3f}°')
            except ValueError:
                st.warning('Selected (hkl) does not satisfy the Bragg condition for the current wavelength.')


    st.divider()
    st.subheader('Fourier Transform — FFT')
    f1, f2 = st.columns([1.4, 1])
    with f1:
        fig = plt.figure(figsize=(6, 5))
        plot_fft_crystal_projection(sim_structure, fig.add_subplot(111), grid_size=fft_grid, sigma_pixels=fft_sigma)
        st.pyplot(fig)
        plt.close(fig)
    with f2:
        st.markdown('**Real space → Fourier space**')
        st.latex(r'F(q_x,q_y)=\mathcal{F}\{\rho(x,y)\}')
        st.latex(r'I_{FFT}(q_x,q_y)=|F(q_x,q_y)|^2')
        st.write('Periodic features in the projected atomic density generate structured features in Fourier space. The q-axes are in Å⁻¹.')

    st.divider()
    st.subheader('Inverse Fourier Transform — IFT')
    st.caption(f'Low-pass reconstruction using |q| ≤ {ift_cutoff:.1f} Å⁻¹.')
    i1, i2 = st.columns(2)
    with i1:
        density, F, F_filtered, reconstruction, qx, qy = fft_ift_reconstruction(
            sim_structure, fft_grid, fft_sigma, ift_cutoff
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(density, origin='lower', aspect='auto')
        ax.set_title('Original Projected Atomic Density')
        ax.set_xlabel('grid x')
        ax.set_ylabel('grid y')
        st.pyplot(fig)
        plt.close(fig)
    with i2:
        fig = plt.figure(figsize=(6, 5))
        plot_ift_reconstruction(sim_structure, fig.add_subplot(111), fft_grid, fft_sigma, ift_cutoff)
        st.pyplot(fig)
        plt.close(fig)
    st.latex(r'\rho_{reconstructed}(x,y)=\mathcal{F}^{-1}\{F(q_x,q_y)\,M(q_x,q_y)\}')
    st.caption('Changing the cutoff demonstrates how removing high spatial frequencies changes the reconstructed real-space image.')
    st.divider()
    st.subheader('FFT Phase & Autocorrelation')
    p1, p2 = st.columns(2)
    with p1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_fft_phase(sim_structure, ax, fft_grid, fft_sigma)
        st.pyplot(fig)
        plt.close(fig)
    with p2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_fft_autocorrelation(sim_structure, ax, fft_grid, fft_sigma)
        st.pyplot(fig)
        plt.close(fig)
    st.caption('The FFT magnitude describes spatial-frequency strength; the phase retains positional information. The autocorrelation reveals characteristic real-space periodicities.');


    st.divider()
    st.subheader('Radial Fourier Profile')
    q, radial = radial_fft_profile(sim_structure, fft_grid, fft_sigma)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(q, radial)
    ax.set_xlabel(r'|q| (Å$^{-1}$)')
    ax.set_ylabel('Mean normalized FFT intensity')
    ax.set_title('Azimuthally Averaged Fourier Spectrum')
    ax.grid(alpha=.2)
    st.pyplot(fig)
    plt.close(fig)

with tab_xrd:
    st.subheader('Simulated Powder XRD')
    tt, I, peaks = powder_pattern(sim_structure, wavelength, hmax=hmax, B=B, peak_width=width)
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_xrd(tt, I, ax)
    st.pyplot(fig)
    plt.close(fig)

    scan_df = pd.DataFrame({'2θ (deg)': tt, 'Intensity (normalized)': I})
    m1, m2, m3 = st.columns(3)
    m1.metric('Simulation points', len(scan_df))
    m2.metric('2θ step', f'{tt[1]-tt[0]:.4f}°')
    m3.metric('Reflections', len(peaks))
    st.download_button('Download simulated XRD CSV', scan_df.to_csv(index=False).encode(), 'simulated_xrd_scan.csv', 'text/csv')

    if peaks:
        peak_df = pd.DataFrame(peaks, columns=['2θ', 'Intensity', 'hkl', 'd (Å)']).sort_values('Intensity', ascending=False)
        st.subheader('Indexed Reflection Analysis')
        peak_df['Bragg check (°)'] = peak_df['2θ']
        peak_df['Relative intensity (%)'] = 100*peak_df['Intensity']/max(peak_df['Intensity'].max(), 1e-15)
        st.dataframe(peak_df.head(20), use_container_width=True)
        st.caption('The reflection table links simulated 2θ, d-spacing, hkl and relative intensity.')

    st.info('The powder-XRD calculation is kept separate from the FFT panel: FFT demonstrates Fourier-space structure, while this panel calculates crystallographic scattering and powder diffraction.')

with tab_exp:
    st.subheader('Experimental XRD')
    up = st.file_uploader('Upload experimental XRD CSV/XY: raw scan or peak table', type=['csv', 'xy', 'txt'], key='exp')
    if up:
        try:
            x, y, xname, yname, meta = load_xrd_csv(up)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Data points', meta['n_points'])
            m2.metric('2θ range', f"{meta['x_min']:.2f}–{meta['x_max']:.2f}°")
            m3.metric('Median Δ2θ', f"{meta['median_step']:.4g}°" if np.isfinite(meta['median_step']) else '—')
            m4.metric('Detected format', 'Peak table' if meta['kind'] == 'peak_table' else 'Raw scan')
            st.caption(f'Using **{xname}** as 2θ and **{yname}** as intensity.')

            # Optional transparent preprocessing of raw scans.
            preprocess = st.expander('Raw-scan preprocessing', expanded=False)
            if meta['kind'] == 'raw_scan':
                with preprocess:
                    use_baseline = st.checkbox('Subtract smooth baseline', value=False)
                    normalize_exp = st.checkbox('Normalize intensity to 0–1', value=False)
                    smooth_exp = st.checkbox('Savitzky–Golay smoothing', value=False)
                    smooth_window = st.slider('Smoothing window (points)', 5, 101, 11, 2)
                    smooth_poly = st.slider('Polynomial order', 2, 4, 2)
                y_proc = np.asarray(y, dtype=float).copy()
                if smooth_exp and smooth_window <= len(y_proc):
                    if smooth_window % 2 == 0:
                        smooth_window += 1
                    smooth_window = min(smooth_window, len(y_proc) if len(y_proc)%2==1 else len(y_proc)-1)
                    if smooth_window > smooth_poly:
                        y_proc = savgol_filter(y_proc, smooth_window, smooth_poly)
                if use_baseline:
                    baseline_window = max(11, min(301, len(y_proc)//10*2+1))
                    if baseline_window % 2 == 0:
                        baseline_window += 1
                    baseline = savgol_filter(y_proc, baseline_window, 2)
                    y_proc = y_proc - baseline
                if normalize_exp:
                    ymin, ymax = np.min(y_proc), np.max(y_proc)
                    y_proc = (y_proc-ymin)/(ymax-ymin+1e-15)
                y = y_proc

            if meta['kind'] == 'peak_table':
                st.warning('This is a peak/reflection table, not a raw XRD scan. Peak detection, FWHM fitting, Scherrer and Williamson–Hall require the original dense scan.')
                table = meta['table'].copy()
                st.dataframe(table, use_container_width=True)
                st.download_button('Download interpreted peak table', table.to_csv(index=False).encode(), 'interpreted_peak_table.csv', 'text/csv')
                a_guess = st.number_input('Lattice parameter guess a (Å)', value=5.4300, min_value=0.1, key='peak_a')
                centering = st.selectbox('Lattice centering', ['primitive', 'fcc', 'bcc'], key='peak_centering')
                tol = st.number_input('Indexing tolerance (°)', value=0.25, min_value=0.01, key='peak_tol')
                idx = index_cubic_peaks(x, a_guess, wavelength, 8, tol, centering)
                idf = pd.DataFrame([{**z, 'hkl': str(z['hkl'])} for z in idx])
                st.dataframe(idf, use_container_width=True)
                st.download_button('Export indexed peaks CSV', idf.to_csv(index=False).encode(), 'indexed_peaks.csv', 'text/csv')
                st.session_state.pop('experimental_analysis', None)
            else:
                if meta['n_points'] < 1000:
                    st.error(f"Raw XRD analysis requires at least **1000 data points**. This file contains **{meta['n_points']}** points.")
                    st.session_state.pop('experimental_analysis', None)
                else:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(x, y, label='Experimental')
                    ax.set_xlabel('2θ (degrees)')
                    ax.set_ylabel('Intensity')
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.subheader('Peak Detection & Fitting')
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        prominence = st.slider('Peak prominence', 0.01, 0.50, 0.05, 0.01)
                    with c2:
                        distance = st.slider('Minimum point distance', 1, 50, 8)
                    with c3:
                        fit_window = st.number_input('Peak fitting window ±°', value=0.50, min_value=0.05, step=0.05)

                    peaks_x, peaks_y, _ = detect_peaks(x, y, prominence, distance)
                    st.metric('Detected peaks', len(peaks_x))
                    st.dataframe(pd.DataFrame({'2θ (deg)': peaks_x, 'Intensity': peaks_y}), use_container_width=True)

                    fit_rows, fit_windows = [], []
                    for center in peaks_x:
                        mask = (x >= center-fit_window) & (x <= center+fit_window)
                        if mask.sum() < 8:
                            continue
                        try:
                            fit = fit_single_peak(x[mask], y[mask], center_guess=float(center), model='pseudo_voigt')
                            if fit['success'] and np.isfinite(fit['fwhm']) and fit['fwhm'] > 0:
                                fit_rows.append({'2θ (deg)': fit['center'], 'FWHM (deg)': fit['fwhm'], 'Intensity': fit['amplitude'], 'R²': fit['r2'], 'η': fit['eta']})
                                fit_windows.append((x[mask], y[mask], fit))
                        except Exception:
                            continue

                    fit_df = pd.DataFrame(fit_rows)
                    if fit_df.empty:
                        st.session_state.pop('experimental_analysis', None)
                        st.warning('No peaks could be fitted reliably with the current settings.')
                    else:
                        fit_df = fit_df.sort_values('2θ (deg)').reset_index(drop=True)
                        st.dataframe(fit_df, use_container_width=True)
                        st.download_button('Export fitted peak table CSV', fit_df.to_csv(index=False).encode(), 'fitted_experimental_peaks.csv', 'text/csv')
                        st.session_state['experimental_analysis'] = {
                            'two_theta': fit_df['2θ (deg)'].to_numpy(),
                            'fwhm': fit_df['FWHM (deg)'].to_numpy(),
                            'intensity': fit_df['Intensity'].to_numpy(),
                            'r2': fit_df['R²'].to_numpy(),
                            'n_points': meta['n_points']
                        }
                        st.success(f'{len(fit_df)} experimental peaks were fitted. These results feed the Size / Strain section.')

                        if fit_windows:
                            plot_peak = st.selectbox('View fitted peak', list(range(len(fit_windows))), format_func=lambda i: f"Peak {i+1}: {fit_windows[i][2]['center']:.3f}°")
                            px, py, pf = fit_windows[plot_peak]
                            fig, ax = plt.subplots(figsize=(8, 3))
                            ax.plot(px, py, '.', label='Experimental')
                            ax.plot(px, pseudo_voigt(px, pf['amplitude'], pf['center'], pf['fwhm'], pf['eta'], pf['background']), label='Pseudo-Voigt fit')
                            ax.set_xlabel('2θ (degrees)')
                            ax.set_ylabel('Intensity')
                            ax.legend()
                            st.pyplot(fig)
                            plt.close(fig)
        except Exception as e:
            st.error(f'XRD analysis error: {e}')
    else:
        st.info('Upload a raw experimental XRD scan to activate peak detection and fitting.')

with tab_size:
    st.subheader('Size / Strain Analysis')
    analysis = st.session_state.get('experimental_analysis')
    if not analysis:
        st.info('No experimental peak-fit results are available. Complete the Experimental XRD section first.')
    else:
        obs_tt = np.asarray(analysis['two_theta'], float)
        obs_fwhm = np.asarray(analysis['fwhm'], float)
        corrected_obs_tt = inst.corrected_two_theta(obs_tt)
        inst_fwhm = inst.instrumental_fwhm(corrected_obs_tt)
        sample_fwhm = correct_instrumental_broadening(obs_fwhm, inst_fwhm)
        valid = sample_fwhm > 0
        if valid.sum() < 2:
            st.error('Instrumental broadening is equal to or larger than the fitted peak widths. Adjust the Caglioti U/V/W model.')
        else:
            theta = corrected_obs_tt[valid] / 2
            corrected_tt = corrected_obs_tt[valid]
            corrected_fwhm = sample_fwhm[valid]
            sizes = scherrer_size(corrected_fwhm, theta, wavelength)
            wh = williamson_hall(corrected_tt, corrected_fwhm, wavelength)
            dframe = pd.DataFrame({
                '2θ (deg)': corrected_tt,
                'Observed FWHM (deg)': obs_fwhm[valid],
                'Instrument FWHM (deg)': inst_fwhm[valid],
                'Corrected Sample FWHM (deg)': corrected_fwhm,
                'Scherrer D (Å)': sizes
            })
            st.dataframe(dframe, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric('Mean Scherrer size', f'{np.mean(sizes):.2f} Å')
            c2.metric('Williamson–Hall size', f'{wh[4]:.2f} Å')
            c3.metric('W–H microstrain', f'{wh[2]:.6g}')
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(wh[0], wh[1], label='Experimental fitted peaks')
            ax.plot(wh[0], wh[3] + wh[2]*wh[0], label='Linear W–H fit')
            ax.set_xlabel('4 sin θ')
            ax.set_ylabel('β cos θ (rad)')
            ax.set_title('Williamson–Hall Size/Strain Analysis')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            reg = linregress(wh[0], wh[1])
            st.metric('Williamson–Hall R²', f'{reg.rvalue**2:.5f}')
            residual = wh[1] - (reg.intercept + reg.slope*wh[0])
            if len(residual) > 2:
                stderr = np.sqrt(np.sum(residual**2)/(len(residual)-2))
                st.caption(f'W–H regression residual standard error: {stderr:.3e} rad')
            st.download_button('Export size/strain report CSV', dframe.to_csv(index=False).encode(), 'size_strain_report.csv', 'text/csv')

with tab_dyn:
    st.subheader('Diatomic Chain: Acoustic + Optical Modes')
    d1, d2 = st.columns([1, 2])
    with d1:
        kappa = st.slider('Nearest-neighbour coupling κ', 0.1, 10.0, 1.0, 0.1)
        m1 = st.slider('Mass M₁', 0.1, 10.0, 1.0, 0.1)
        m2 = st.slider('Mass M₂', 0.1, 10.0, 2.0, 0.1)
        st.latex(r'\omega_{\pm}^2=\kappa(1/M_1+1/M_2)\pm\sqrt{\kappa^2(1/M_1+1/M_2)^2-4\kappa^2\sin^2(qa/2)/(M_1M_2)}')
    with d2:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_diatomic_dispersion(ax, kappa, m1, m2)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader('Lattice Dynamics — 1D Harmonic Chain')
    st.caption('A transparent monoatomic nearest-neighbour model for visualizing acoustic phonons. This is a model dispersion, not a first-principles phonon calculation for the uploaded material.')
    c1, c2 = st.columns([1, 2])
    with c1:
        spring_constant = st.slider('Spring constant K (normalized)', 0.1, 10.0, 1.0, 0.1)
        mass = st.slider('Atomic mass M (normalized)', 0.1, 10.0, 1.0, 0.1)
        sound_speed = np.sqrt(spring_constant/mass)
        st.metric('Long-wavelength slope', f'{sound_speed:.3f} (normalized)')
        st.latex(r'\omega(q)=2\sqrt{K/M}\,|\sin(qa/2)|')
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_phonon_dispersion(ax, spring_constant, mass)
        st.pyplot(fig)
        plt.close(fig)

st.divider()
st.markdown("<div style='text-align:center; padding:18px 0 6px; color:#666; font-size:0.9rem;'>© 2026 Aman Kumar Patel · Made with ❤️ by Aman Kumar Patel</div>", unsafe_allow_html=True)
st.caption('Research note: It is a transparent computational workstation, not a validated Rietveld or first-principles phonon package.')
