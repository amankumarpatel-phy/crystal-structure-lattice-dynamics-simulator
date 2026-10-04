import os, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from crystal.cif import read_cif
from crystal.scattering import powder_pattern
from visualization.plots import plot_structure, plot_xrd, plot_reciprocal_lattice, plot_fft_crystal_projection
from analysis.io import load_xrd_csv
from analysis.xrd import detect_peaks, correct_instrumental_broadening, scherrer_size, williamson_hall
from analysis.refinement import fit_single_peak, pseudo_voigt
from analysis.instrument import InstrumentConfig
from analysis.indexing import index_cubic_peaks

st.set_page_config(page_title='Crystal Structure & Lattice Dynamics Simulator V5',layout='wide')
st.title('Crystal Structure & Lattice Dynamics Simulator — V5')
st.caption('CIF → reciprocal space → XRD → experimental peak analysis')

with st.sidebar:
    st.header('Instrument / Simulation')
    wavelength=st.number_input('X-ray wavelength λ (Å)',value=1.5406,min_value=0.1)
    hmax=st.slider('Maximum |h,k,l|',1,8,5)
    B=st.number_input('Debye–Waller B (Å²)',value=0.0,min_value=0.0)
    width=st.number_input('Simulated Gaussian width (°)',value=0.10,min_value=0.01)
    reciprocal_hmax=st.slider('Reciprocal lattice |h,k,l|',1,6,4)
    plane_h=st.number_input('Reciprocal plane h',value=1,min_value=-6,max_value=6,step=1)
    plane_k=st.number_input('Reciprocal plane k',value=0,min_value=-6,max_value=6,step=1)
    plane_l=st.number_input('Reciprocal plane l',value=0,min_value=-6,max_value=6,step=1)
    fft_grid=st.select_slider('FFT grid size',[64,128,256,512],value=256)
    fft_sigma=st.slider('FFT atomic-density smoothing (pixels)',0.0,3.0,1.0,0.25)
    U=st.number_input('Caglioti U',value=0.0,format='%.6f')
    V=st.number_input('Caglioti V',value=0.0,format='%.6f')
    W=st.number_input('Caglioti W (deg²)',value=0.01,min_value=0.0,format='%.6f')
    zero=st.number_input('2θ zero shift (°)',value=0.0,format='%.4f')
    cif=st.file_uploader('Upload CIF file',type=['cif'])

inst=InstrumentConfig(wavelength_A=wavelength,U=U,V=V,W=W,zero_shift_deg=zero)
if cif:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.cif') as f:
        f.write(cif.getbuffer()); path=f.name
    try: s=read_cif(path)
    finally: os.unlink(path)
else:
    s=read_cif('data/Si.cif')

tab1,tab2,tab3=st.tabs(['Structure & Simulation','Experimental XRD','Size / Strain'])

with tab1:
    c1,c2=st.columns(2)
    with c1:
        fig=plt.figure(figsize=(6,5)); plot_structure(s,fig.add_subplot(111,projection='3d')); st.pyplot(fig); plt.close(fig)
        st.metric('Unit-cell volume',f'{s.volume:.3f} Å³')
        a,b,c=np.linalg.norm(s.cell[0]),np.linalg.norm(s.cell[1]),np.linalg.norm(s.cell[2])
        st.write(f'a={a:.4f} Å, b={b:.4f} Å, c={c:.4f} Å')
    with c2:
        fig=plt.figure(figsize=(6,5))
        plot_reciprocal_lattice(s,fig.add_subplot(111,projection='3d'),hmax=reciprocal_hmax,plane_hkl=(plane_h,plane_k,plane_l))
        st.pyplot(fig); plt.close(fig)
        st.caption(f'Reciprocal-space visualization: lattice points up to |h,k,l| ≤ {reciprocal_hmax}; selected plane ({plane_h}{plane_k}{plane_l}).')
    st.subheader('Fourier Transform / FFT')
    st.caption('2D FFT of the projected atomic density. This is a Fourier-space visualization and is intentionally shown separately from the powder-XRD scattering model.')
    f1,f2=st.columns(2)
    with f1:
        fig=plt.figure(figsize=(6,5))
        plot_fft_crystal_projection(s,fig.add_subplot(111),grid_size=fft_grid,sigma_pixels=fft_sigma)
        st.pyplot(fig); plt.close(fig)
    with f2:
        st.markdown('**Fourier-space interpretation**')
        st.latex(r'F(q_x,q_y)=\\mathcal{F}\\{\\rho(x,y)\\}')
        st.latex(r'I_{FFT}(q_x,q_y)=|F(q_x,q_y)|^2')
        st.write('Periodic features in the projected real-space crystal produce structured features in reciprocal/Fourier space. The displayed q-axis is in Å⁻¹.')
    c1,c2=st.columns(2)
    with c1:
        tt,I,peaks=powder_pattern(s,wavelength,hmax=hmax,B=B,peak_width=width)
        fig,ax=plt.subplots(figsize=(7,4)); plot_xrd(tt,I,ax); st.pyplot(fig); plt.close(fig)

        # The simulated pattern is a dense scan (default: 3601 points).
        scan_df=pd.DataFrame({'2θ (deg)':tt,'Intensity (normalized)':I})
        st.subheader('Simulated XRD scan')
        st.caption(f'Dense simulated dataset: **{len(scan_df)} points** with Δ2θ ≈ **{(tt[1]-tt[0]):.4f}°**.')
        st.dataframe(scan_df,use_container_width=True,height=320)
        st.download_button(
            'Download simulated XRD CSV',
            scan_df.to_csv(index=False).encode(),
            'simulated_xrd_scan.csv',
            'text/csv'
        )

        if peaks:
            peak_df=pd.DataFrame(peaks,columns=['2θ','Intensity','hkl','d (Å)']).sort_values('Intensity',ascending=False)
            st.subheader('Reflection summary')
            st.caption('Showing the 15 strongest calculated reflections; the full scan above contains all simulated intensity points.')
            st.dataframe(peak_df.head(15),use_container_width=True)

with tab2:
    up=st.file_uploader('Upload experimental XRD CSV/XY: raw scan or peak table',type=['csv','xy','txt'],key='exp')
    if up:
        try:
            x,y,xname,yname,meta=load_xrd_csv(up)

            st.subheader('Data diagnostics')
            m1,m2,m3,m4=st.columns(4)
            m1.metric('Data points',meta['n_points'])
            m2.metric('2θ range',f"{meta['x_min']:.2f}–{meta['x_max']:.2f}°")
            m3.metric('Median Δ2θ',f"{meta['median_step']:.4g}°" if np.isfinite(meta['median_step']) else '—')
            m4.metric('Detected format','Peak table' if meta['kind']=='peak_table' else 'Raw scan')
            st.caption(f"Using **{xname}** as 2θ and **{yname}** as intensity.")

            if meta['kind']=='peak_table':
                st.warning(
                    'This file is a diffraction **peak/reflection table**, not a raw XRD scan. '
                    'Peak detection, FWHM fitting, Scherrer analysis and Williamson–Hall analysis '
                    'require the original dense experimental 2θ–intensity scan.'
                )
                table=meta['table'].copy()
                st.dataframe(table,use_container_width=True)
                st.download_button('Download interpreted peak table',table.to_csv(index=False).encode(),'interpreted_peak_table.csv','text/csv')
                st.subheader('Reflection / peak table')
                a_guess=st.number_input('Lattice parameter guess a (Å)',value=5.4300,min_value=0.1,key='peak_a')
                centering=st.selectbox('Lattice centering',['primitive','fcc','bcc'],key='peak_centering')
                tol=st.number_input('Indexing tolerance (°)',value=0.25,min_value=0.01,key='peak_tol')
                idx=index_cubic_peaks(x,a_guess,wavelength,8,tol,centering)
                idf=pd.DataFrame([{**z,'hkl':str(z['hkl'])} for z in idx])
                st.dataframe(idf,use_container_width=True)
                st.download_button('Export indexed peaks CSV',idf.to_csv(index=False).encode(),'indexed_peaks.csv','text/csv')
                st.session_state.pop('experimental_analysis',None)

            else:
                if meta['n_points'] < 1000:
                    st.error(
                        f"Raw XRD analysis requires at least **1000 data points**. "
                        f"This file contains **{meta['n_points']}** points. "
                        "Please upload the original dense experimental 2θ–intensity scan."
                    )
                    st.session_state.pop('experimental_analysis',None)
                else:
                    fig,ax=plt.subplots(figsize=(9,4))
                    ax.plot(x,y,label='Experimental')
                    ax.set_xlabel('2θ (degrees)'); ax.set_ylabel('Intensity'); ax.legend()
                    st.pyplot(fig); plt.close(fig)

                    st.subheader('Peak detection')
                    c1,c2,c3=st.columns(3)
                    with c1:
                        prominence=st.slider('Peak prominence',0.01,0.50,0.05,0.01)
                    with c2:
                        distance=st.slider('Minimum point distance',1,50,8)
                    with c3:
                        fit_window=st.number_input('Peak fitting window ±°',value=0.50,min_value=0.05,step=0.05)

                    peaks_x,peaks_y,_=detect_peaks(x,y,prominence,distance)
                    st.metric('Detected peaks',len(peaks_x))
                    peak_summary=pd.DataFrame({'2θ (deg)':peaks_x,'Intensity':peaks_y})
                    st.dataframe(peak_summary,use_container_width=True)

                    if len(peaks_x)==0:
                        st.session_state.pop('experimental_analysis',None)
                        st.info('No significant local maxima were detected. Adjust peak prominence/distance or check the raw scan.')
                    else:
                        st.subheader('Automatic peak fitting')
                        fit_rows=[]
                        fit_windows=[]
                        for center in peaks_x:
                            mask=(x>=center-fit_window)&(x<=center+fit_window)
                            if mask.sum()<8:
                                continue
                            try:
                                fit=fit_single_peak(x[mask],y[mask],center_guess=float(center),model='pseudo_voigt')
                                if fit['success'] and np.isfinite(fit['fwhm']) and fit['fwhm']>0:
                                    fit_rows.append({
                                        '2θ (deg)':fit['center'],
                                        'FWHM (deg)':fit['fwhm'],
                                        'Intensity':fit['amplitude'],
                                        'R²':fit['r2'],
                                        'η':fit['eta']
                                    })
                                    fit_windows.append((x[mask],y[mask],fit))
                            except Exception:
                                continue

                        fit_df=pd.DataFrame(fit_rows)
                        if fit_df.empty:
                            st.session_state.pop('experimental_analysis',None)
                            st.warning('No peaks could be fitted reliably with the current fitting window.')
                        else:
                            fit_df=fit_df.sort_values('2θ (deg)').reset_index(drop=True)
                            st.dataframe(fit_df,use_container_width=True)
                            st.download_button(
                                'Export fitted peak table CSV',
                                fit_df.to_csv(index=False).encode(),
                                'fitted_experimental_peaks.csv',
                                'text/csv'
                            )

                            # Store only results derived from the current experimental scan.
                            st.session_state['experimental_analysis']={
                                'two_theta':fit_df['2θ (deg)'].to_numpy(),
                                'fwhm':fit_df['FWHM (deg)'].to_numpy(),
                                'intensity':fit_df['Intensity'].to_numpy(),
                                'r2':fit_df['R²'].to_numpy(),
                                'x_name':xname,
                                'y_name':yname,
                                'n_points':meta['n_points'],
                            }

                            st.success(
                                f"{len(fit_df)} experimental peaks were fitted. "
                                "These fitted values now feed the Size / Strain analysis."
                            )

                            if len(fit_windows):
                                plot_peak=st.selectbox(
                                    'View fitted peak',
                                    list(range(len(fit_windows))),
                                    format_func=lambda i: f"Peak {i+1}: {fit_windows[i][2]['center']:.3f}°"
                                )
                                px,py,pf=fit_windows[plot_peak]
                                fig,ax=plt.subplots(figsize=(8,3))
                                ax.plot(px,py,'.',label='Experimental')
                                ax.plot(
                                    px,
                                    pseudo_voigt(px,pf['amplitude'],pf['center'],pf['fwhm'],pf['eta'],pf['background']),
                                    label='Pseudo-Voigt fit'
                                )
                                ax.set_xlabel('2θ (degrees)'); ax.set_ylabel('Intensity'); ax.legend()
                                st.pyplot(fig); plt.close(fig)

        except Exception as e:
            st.error(f"XRD analysis error: {e}")

with tab3:
    st.subheader('Size / Strain Analysis')
    analysis=st.session_state.get('experimental_analysis')

    if not analysis:
        st.info(
            'No experimental peak-fit results are available yet. '
            'Go to **Experimental XRD**, upload a raw scan with at least 1000 points, '
            'detect the peaks, and let the application fit them first.'
        )
    else:
        obs_tt=np.asarray(analysis['two_theta'],float)
        obs_fwhm=np.asarray(analysis['fwhm'],float)

        st.write(
            f"Using **{len(obs_tt)} fitted experimental peaks** from the current XRD dataset "
            f"({analysis['n_points']} raw data points)."
        )

        # Propagate the instrument 2θ zero-shift into the downstream analysis.
        corrected_obs_tt=inst.corrected_two_theta(obs_tt)
        inst_fwhm=inst.instrumental_fwhm(corrected_obs_tt)
        sample_fwhm=correct_instrumental_broadening(obs_fwhm,inst_fwhm)

        valid=sample_fwhm>0
        if valid.sum()<2:
            st.error(
                'Instrumental broadening is equal to or larger than the fitted peak widths. '
                'Adjust the Caglioti U/V/W parameters or use an instrument profile calibrated from a standard.'
            )
        else:
            theta=corrected_obs_tt[valid]/2
            corrected_tt=corrected_obs_tt[valid]
            corrected_fwhm=sample_fwhm[valid]
            sizes=scherrer_size(corrected_fwhm,theta,wavelength)
            wh=williamson_hall(corrected_tt,corrected_fwhm,wavelength)

            dframe=pd.DataFrame({
                '2θ (deg)':corrected_tt,
                'Observed FWHM (deg)':obs_fwhm[valid],
                'Instrument FWHM (deg)':inst_fwhm[valid],
                'Corrected Sample FWHM (deg)':corrected_fwhm,
                'Scherrer D (Å)':sizes
            })
            st.dataframe(dframe,use_container_width=True)

            c1,c2=st.columns(2)
            with c1:
                st.metric('Mean Scherrer crystallite size',f'{np.mean(sizes):.2f} Å')
            with c2:
                st.metric('Williamson–Hall size',f'{wh[4]:.2f} Å')

            st.metric('Williamson–Hall microstrain',f'{wh[2]:.6g}')

            fig,ax=plt.subplots(figsize=(7,4))
            ax.scatter(wh[0],wh[1],label='Experimental fitted peaks')
            ax.plot(wh[0],wh[3]+wh[2]*wh[0],label='Linear W–H fit')
            ax.set_xlabel('4 sin θ')
            ax.set_ylabel('β cos θ (rad)')
            ax.set_title('Williamson–Hall Size/Strain Analysis')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.download_button(
                'Export size/strain report CSV',
                dframe.to_csv(index=False).encode(),
                'size_strain_report.csv',
                'text/csv'
            )

            st.caption(
                'All values above are derived from the uploaded experimental XRD scan, '
                'the fitted peak widths, the selected wavelength, and the Caglioti U/V/W instrument model.'
            )

st.divider()
st.markdown("<div style='text-align:center; padding:18px 0 6px; color:#666; font-size:0.9rem;'>© 2026 Aman Kumar Patel · Made with ❤️ by Aman Kumar Patel</div>", unsafe_allow_html=True)
st.caption('Research note: V5 is a transparent analysis workstation, not a validated Rietveld refinement package.')
