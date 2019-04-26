import sys, os
import argparse

try:
    sys.path.append(os.environ['SLURM_JOBTMP'] + "/NPTFit/")
except KeyError:
    sys.path.append("../../NPTFit/")

import numpy as np

from NPTFit import nptfit 
from NPTFit import create_mask as cm
from NPTFit import psf_correction as pc

parser = argparse.ArgumentParser()
parser.add_argument("--psf_king", action="store", dest="psf_king", default=0, type=int)
parser.add_argument("--i_mc", action="store", dest="i_mc", default=0, type=int)
parser.add_argument("--i_xsec", action="store", dest="i_xsec", default=0, type=int)
parser.add_argument("--ps_mask", action="store", dest="ps_mask", default="3fgl_0p8deg", type=str)
parser.add_argument("--nexp", action="store", dest="nexp", default=1, type=int)

results=parser.parse_args()

psf_king=results.psf_king
i_mc=results.i_mc
i_xsec=results.i_xsec
ps_mask=results.ps_mask
nexp=results.nexp

i_ebin = 0
work_dir = '/scratch/sm8383/NEXP' + str(nexp) + '/'

tag = 'mc_' + str(i_mc) + "_xsec_" + str(i_xsec) + "_king_" + str(psf_king) + "_mask_" + ps_mask

n = nptfit.NPTF(work_dir=work_dir, tag=tag) 

fermi_data = np.load('../data/MC_inj_data.npy')[i_mc, i_xsec, i_ebin]
fermi_exposure = np.load('../data/fermi_data/fermidata_exposure.npy')

n.load_data(fermi_data, fermi_exposure)

if ps_mask == "3fgl_0p95psf":
    pscmask = np.array(np.load('../data/fermi_data/fermidata_pscmask.npy'), dtype=bool)
elif ps_mask == "3fgl_0p8deg":
    pscmask = np.array(np.load('../data/mask_3fgl_0p8deg.npy'), dtype=bool)
elif ps_mask == "4fgl_0p8deg":
    pscmask = np.array(np.load('../data/mask_4fgl_0p8deg.npy'), dtype=bool)

analysis_mask = cm.make_mask_total(band_mask = True, band_mask_range = 2,
                                   mask_ring = True, inner = 0, outer = 30,
                                   custom_mask = pscmask)

n.load_mask(analysis_mask)

dif = np.load('../data/fermi_data/template_dif.npy')
iso = np.load('../data/fermi_data/template_iso.npy')
bub = np.load('../data/fermi_data/template_bub.npy')
gce = np.load('../data/SGH_Jfactor_map_NFW_gamma_1.2_baseline.npy')
dsk = np.load('../data/fermi_data/template_dsk.npy')
psc = np.load('../data/fermi_data/template_psc.npy')

n.add_template(dif, 'dif')
n.add_template(iso, 'iso')
n.add_template(bub, 'bub')
n.add_template(gce, 'gce')
n.add_template(dsk, 'dsk')
n.add_template(psc, 'psc')

# Remove the exposure correction for PS templates
rescale = fermi_exposure/np.mean(fermi_exposure)
n.add_template(gce/rescale, 'gce_np', units='PS')
n.add_template(dsk/rescale, 'dsk_np', units='PS')
n.add_template(iso/rescale, 'iso_np', units='PS')

n.add_poiss_model('dif', '$A_\mathrm{dif}$', [8,17], False)
n.add_poiss_model('iso', '$A_\mathrm{iso}$', [0,2], False)
n.add_poiss_model('gce', '$A_\mathrm{gce}$', [-4,3], True)
n.add_poiss_model('bub', '$A_\mathrm{bub}$', [0,2], False)
n.add_poiss_model('psc', '$A_\mathrm{psc}$', [0,2], False)

np_priors = [[-6,2],[2.05,15],[-4.95,0.05],[-4.95,0.05],[np.log10(2.01),np.log10(100.)],[np.log10(0.01),np.log10(2.)]]
np_priors_log = [True,False,False,False,True,True]

n.add_non_poiss_model('gce_np',
                        ['$A_\mathrm{gce}^\mathrm{ps}$','$n_1^\mathrm{gce}$','$n_2^\mathrm{gce}$','$n_3^\mathrm{gce}$','$S_b^{(1), \mathrm{gce}}$','$S_b^{(2), \mathrm{gce}}$'],
                        np_priors,
                        np_priors_log)

# n.add_non_poiss_model('dsk_np',
#                       ['$A_\mathrm{dsk}^\mathrm{ps}$','$n_1^\mathrm{dsk}$','$n_2^\mathrm{dsk}$','$n_3^\mathrm{dsk}$','$S_b^{(1), \mathrm{dsk}}$','$S_b^{(2), \mathrm{dsk}}$'],
#                       np_priors,
#                       np_priors_log)

n.add_non_poiss_model('iso_np',
                      ['$A_\mathrm{iso}^\mathrm{ps}$','$n_1^\mathrm{iso}$','$n_2^\mathrm{iso}$','$n_3^\mathrm{iso}$','$S_b^{(1), \mathrm{iso}}$','$S_b^{(2), \mathrm{iso}}$'],
                      np_priors,
                      np_priors_log)

if psf_king:

    # Define parameters that specify the Fermi-LAT PSF at 2 GeV
    fcore = 0.748988248179
    score = 0.428653790656
    gcore = 7.82363229341
    stail = 0.715962650769
    gtail = 3.61883748683
    spe = 0.00456544262478

    # Define the full PSF in terms of two King functions
    def king_fn(x, sigma, gamma):
        return 1./(2.*np.pi*sigma**2.)*(1.-1./gamma)*(1.+(x**2./(2.*gamma*sigma**2.)))**(-gamma)

    def Fermi_PSF(r):
        return fcore*king_fn(r/spe,score,gcore) + (1-fcore)*king_fn(r/spe,stail,gtail)

    # Modify the relevant parameters in pc_inst and then make or load the PSF
    pc_inst = pc.PSFCorrection(delay_compute=True)
    pc_inst.psf_r_func = lambda r: Fermi_PSF(r)
    pc_inst.sample_psf_max = 10.*spe*(score+stail)/2.
    pc_inst.psf_samples = 10000
    pc_inst.psf_tag = 'Fermi_PSF_2GeV'
    pc_inst.make_or_load_psf_corr()

    # Extract f_ary and df_rho_div_f_ary as usual
    f_ary = pc_inst.f_ary
    df_rho_div_f_ary = pc_inst.df_rho_div_f_ary

else:

    pc_inst = pc.PSFCorrection(psf_sigma_deg=0.1812)
    f_ary, df_rho_div_f_ary = pc_inst.f_ary, pc_inst.df_rho_div_f_ary

n.configure_for_scan(f_ary, df_rho_div_f_ary, nexp=nexp)

n.perform_scan(nlive=100)