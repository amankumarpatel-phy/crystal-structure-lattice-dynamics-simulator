import os, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from crystal.cif import read_cif
from crystal.scattering import powder_pattern
from visualization.plots import plot_structure, plot_xrd
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
        tt,I,peaks=powder_pattern(s,wavelength,hmax=hmax,B=B,peak_width=width)
        fig,ax=plt.subplots(figsize=(7,4)); plot_xrd(tt,I,ax); st.pyplot(fig); plt.close(fig)
        if peaks:
            df=pd.DataFrame(peaks,columns=['2θ','Intensity','hkl','d (Å)']).sort_values('Intensity',ascending=False).head(15)
            st.dataframe(df,use_container_width=True)

with tab2:
    up=st.file_uploader('Upload experimental XRD CSV/XY: raw scan or peak table',type=['csv','xy','txt'],key='exp')
    if up:
        try:
            x,y,xname,yname,meta=load_xrd_csv(up)

            # Show exactly how the uploaded file was interpreted.
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
                    'Peak detection and FWHM fitting are disabled for this file. '
                    'For peak fitting, upload the original dense experimental 2θ–intensity scan.'
                )
                table=meta['table'].copy()
                st.dataframe(table,use_container_width=True)
                st.download_button('Download interpreted peak table',table.to_csv(index=False).encode(),'interpreted_peak_table.csv','text/csv')

                # A peak table can still be useful for indexing.
                st.subheader('Reflection / peak table')
                a_guess=st.number_input('Lattice parameter guess a (Å)',value=5.4300,min_value=0.1,key='peak_a')
                centering=st.selectbox('Lattice centering',['primitive','fcc','bcc'],key='peak_centering')
                tol=st.number_input('Indexing tolerance (°)',value=0.25,min_value=0.01,key='peak_tol')
                idx=index_cubic_peaks(x,a_guess,wavelength,8,tol,centering)
                idf=pd.DataFrame([{**z,'hkl':str(z['hkl'])} for z in idx])
                st.dataframe(idf,use_container_width=True)
                st.download_button('Export indexed peaks CSV',idf.to_csv(index=False).encode(),'indexed_peaks.csv','text/csv')
            else:
                # Raw scan: this is the only format passed to peak detection.
                fig,ax=plt.subplots(figsize=(9,4))
                ax.plot(x,y,label='Experimental')
                ax.set_xlabel('2θ (degrees)'); ax.set_ylabel('Intensity'); ax.legend()
                st.pyplot(fig); plt.close(fig)

                prominence=st.slider('Peak prominence',0.01,0.50,0.05,0.01)
                distance=st.slider('Minimum point distance',1,50,8)
                peaks_x,peaks_y,_=detect_peaks(x,y,prominence,distance)
                st.metric('Detected peaks',len(peaks_x))
                st.dataframe(pd.DataFrame({'2θ (deg)':peaks_x,'Intensity':peaks_y}),use_container_width=True)

                if len(peaks_x)==0:
                    st.info('No significant local maxima were detected. Check the raw scan, baseline/background, and peak-prominence setting.')
                else:
                    a_guess=st.number_input('Lattice parameter guess a (Å)',value=5.4300,min_value=0.1,key='raw_a')
                    centering=st.selectbox('Lattice centering',['primitive','fcc','bcc'],key='raw_centering')
                    tol=st.number_input('Indexing tolerance (°)',value=0.25,min_value=0.01,key='raw_tol')
                    idx=index_cubic_peaks(peaks_x,a_guess,wavelength,8,tol,centering)
                    idf=pd.DataFrame([{**z,'hkl':str(z['hkl'])} for z in idx])
                    st.dataframe(idf,use_container_width=True)
                    st.download_button('Export indexed peaks CSV',idf.to_csv(index=False).encode(),'indexed_peaks.csv','text/csv')

                    center=st.number_input('Peak center guess 2θ (deg)',value=float(peaks_x[0]))
                    window=st.number_input('Fit window ±°',value=0.5,min_value=0.05)
                    mask=(x>=center-window)&(x<=center+window)
                    if mask.sum() >= 8:
                        fit=fit_single_peak(x[mask],y[mask],center_guess=center,model='pseudo_voigt')
                        st.json(fit)
                        fig,ax=plt.subplots(figsize=(8,3))
                        ax.plot(x[mask],y[mask],'.',label='data')
                        ax.plot(x[mask],pseudo_voigt(x[mask],fit['amplitude'],fit['center'],fit['fwhm'],fit['eta'],fit['background']),label='pseudo-Voigt')
                        ax.legend(); ax.set_xlabel('2θ'); ax.set_ylabel('Intensity')
                        st.pyplot(fig); plt.close(fig)
                    else:
                        st.warning('The selected peak window contains too few points for reliable profile fitting.')
        except Exception as e:
            st.error(str(e))
    else:
        st.info('Upload experimental data to activate peak detection, indexing and profile fitting.')

with tab3:
    st.write('Enter fitted peak positions and observed FWHM values.')
    tt_text=st.text_input('2θ peaks (comma separated)','28.44,47.30,56.12')
    fwhm_text=st.text_input('Observed FWHM (deg, comma separated)','0.16,0.20,0.24')
    try:
        tt=np.array([float(v.strip()) for v in tt_text.split(',')]); fwhm=np.array([float(v.strip()) for v in fwhm_text.split(',')])
        if len(tt)!=len(fwhm): raise ValueError('Peak and FWHM counts must match.')
        inst_fwhm=inst.instrumental_fwhm(tt); sample_fwhm=correct_instrumental_broadening(fwhm,inst_fwhm)
        sizes=scherrer_size(sample_fwhm,tt/2,wavelength); wh=williamson_hall(tt,sample_fwhm,wavelength)
        dframe=pd.DataFrame({'2θ (deg)':tt,'Observed FWHM (deg)':fwhm,'Instrument FWHM (deg)':inst_fwhm,'Sample FWHM (deg)':sample_fwhm,'Scherrer D (Å)':sizes})
        st.dataframe(dframe,use_container_width=True)
        st.metric('Williamson–Hall size',f'{wh[4]:.2f} Å'); st.metric('Williamson–Hall microstrain',f'{wh[2]:.5g}')
        fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(wh[0],wh[1]); ax.plot(wh[0],wh[3]+wh[2]*wh[0]); ax.set_xlabel('4 sin θ'); ax.set_ylabel('β cos θ (rad)'); ax.set_title('Williamson–Hall'); st.pyplot(fig); plt.close(fig)
        st.download_button('Export size/strain report CSV',dframe.to_csv(index=False).encode(),'size_strain_report.csv','text/csv')
    except Exception as e: st.error(str(e))

st.divider()
st.markdown("<div style='text-align:center; padding:18px 0 6px; color:#666; font-size:0.9rem;'>© 2026 Aman Kumar Patel · Made with ❤️ by Aman Kumar Patel</div>", unsafe_allow_html=True)
st.caption('Research note: V5 is a transparent analysis workstation, not a validated Rietveld refinement package.')
