import sys, os, argparse, ast
import numpy as np

from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis # module for analysing the output

# Parse input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--mask_band", action="store", dest="mask_band", default=0,type=int)
parser.add_argument("--mask_bandval", action="store", dest="mask_bandval", default=30,type=int)
parser.add_argument("--mask_ring", action="store", dest="mask_ring", default=0,type=int)
parser.add_argument("--mask_innerval", action="store", dest="mask_innerval", default=0,type=int)
parser.add_argument("--mask_outerval", action="store", dest="mask_outerval", default=45,type=int)
parser.add_argument("--mask_b", action="store", dest="mask_b", default=0,type=int)
parser.add_argument("--mask_bmin", action="store", dest="mask_bmin", default=0,type=int)
parser.add_argument("--mask_bmax", action="store", dest="mask_bmax", default=0,type=int)
parser.add_argument("--mask_l", action="store", dest="mask_l", default=0,type=int)
parser.add_argument("--mask_lmin", action="store", dest="mask_lmin", default=0,type=int)
parser.add_argument("--mask_lmax", action="store", dest="mask_lmax", default=0,type=int)
parser.add_argument("--model_GCE", action="store", dest="model_GCE", default=0,type=int)
parser.add_argument("--model_PSF", action="store", dest="model_PSF", default=0,type=int)
parser.add_argument("--psf_king", action="store", dest="psf_king", default=0,type=int)
parser.add_argument("--scan_data", action="store", dest="scan_data", default=0,type=int)
parser.add_argument("--run_tag", action="store", dest="run_tag", default="",type=str)
parser.add_argument('--data_file_path',action='store', dest='data_file_path', default='',type=str)
parser.add_argument('--data_dir',action='store', dest='data_dir', default='/tigress/ljchang/NPTF-IG-Check/Bkg-Maps/fermi_data/',type=str)
parser.add_argument('--work_dir',action='store', dest='work_dir', default='/tigress/ljchang/NPTF-IG-Check/chains/',type=str)

results = parser.parse_args()
mask_band = results.mask_band
mask_bandval = results.mask_bandval
mask_ring = results.mask_ring
mask_outerval = results.mask_outerval
mask_innerval = results.mask_innerval
mask_b = results.mask_b
mask_bmin = results.mask_bmin
mask_bmax = results.mask_bmax
mask_l = results.mask_l
mask_lmin = results.mask_lmin
mask_lmax = results.mask_lmax
model_GCE = results.model_GCE
model_PSF = results.model_PSF
psf_king = results.psf_king
scan_data = results.scan_data
run_tag = results.run_tag
data_file_path = results.data_file_path
data_dir = results.data_dir
work_dir = results.work_dir

mask_options_dict={"band_mask":mask_band,"band_mask_range":mask_bandval,"mask_ring":mask_ring,"inner":mask_innerval,"outer":mask_outerval,
"b_mask":mask_b,"b_deg_min":mask_bmin,"b_deg_max":mask_bmax,"l_mask":mask_l,"l_deg_min":mask_lmin,"l_deg_max":mask_lmax}

analysis_mask_base = cm.make_mask_total(**mask_options_dict)

############################
# Load data  and templates #
############################
templates_dir = "/tigress/ljchang/NPTF-IG-Check/Bkg-Maps/templates_example/ps_mask_0.8deg/"
best_fit_norms = np.load(templates_dir+"best_fit_norms_band_2_ring_30.npy")

fermi_data = np.load(data_dir+'fermidata_counts.npy').astype(np.int32)
if not scan_data:
	data = np.load(data_file_path).astype(np.int32)
	# data = np.load(data_file_path)['counts_map'].astype(np.int32)
fermi_exposure = np.load(data_dir+'fermidata_exposure.npy')
dif = np.load(data_dir+'template_dif.npy')
p8 = np.load("/tigress/smsharma/Fermi-SmoothGalHalo/DataFiles/Bkg-Maps/fermi_data/eventclass5_eventtype3/dif_p8_ebin_10.npy")
iso = np.load(data_dir+'template_iso.npy')
psc = np.load(data_dir+'template_psc.npy')
bub = np.load(data_dir+'template_bub.npy')
dsk = np.load(data_dir+'template_dsk.npy')
nfw = np.load('/tigress/smsharma/Fermi-SmoothGalHalo/DataFiles/SGH_Jfactor_map_NFW_gamma_1.2.npy')*fermi_exposure 
nfw_smoothed = np.load('/tigress/ljchang/NPTF-IG-Check/Bkg-Maps/JfactorSmoothed/SGH_Jfactor_map_NFW_gamma_1.2_smoothed.npy')[10]*fermi_exposure
ps_mask = np.load('/tigress/ljchang/NPTF-IG-Check/data/mask_3fgl_0p8deg.npy')

############################
# Set up NPTF scan #
############################

n = nptfit.NPTF(work_dir=work_dir,tag=run_tag)
exposure_map = np.ones(len(fermi_exposure))*np.mean(fermi_exposure)
if not scan_data:
	n.load_data(data, exposure_map)
	# n.load_data(data, fermi_exposure)
	print("Running on MC")
else:
	n.load_data(fermi_data, exposure_map)
	# n.load_data(data, fermi_exposure)

	print("Running on data")

analysis_mask = analysis_mask_base + ps_mask
analysis_mask = analysis_mask > 0 
n.load_mask(analysis_mask)

# Remove the exposure correction for PS templates
rescale = fermi_exposure/np.mean(fermi_exposure)

# n.add_template(p8, 'dif')
n.add_template(dif/rescale, 'dif')
# n.add_template(dif, 'dif')

# n.add_template(np.ones(len(iso)), 'iso')
n.add_template(iso/rescale, 'iso')
n.add_template(psc/rescale, 'psc')
n.add_template(bub/rescale, 'bub')
n.add_template(nfw_smoothed*0.0002/rescale, 'nfw_dm') # Adjust norm of template such that A=1 corresp. to xsec of ~1e-26

# Adjust PS template norms such that A~1 on data
iso_ps = np.ones(len(iso))
n.add_template(iso_ps*0.07, 'iso_ps',units='PS') 
n.add_template(dsk*0.004/rescale, 'dsk_ps', units='PS') 
n.add_template(nfw*1e-5/rescale, 'nfw_ps', units='PS')

# n.add_poiss_model('dif', '$A_\mathrm{dif}$', [0,20], False)
# # n.add_poiss_model('dif', '$A_\mathrm{dif}$', fixed=True, fixed_norm=best_fit_norms[0])
# n.add_poiss_model('iso', '$A_\mathrm{iso}$', [0,2], False)
# n.add_poiss_model('psc', '$A_\mathrm{psc}$', [0,2], False)
# n.add_poiss_model('bub', '$A_\mathrm{bub}$', [0,2], False)
# n.add_poiss_model('nfw_dm', '$A_\mathrm{nfw_{dm}}$', [0,10], False)

n.add_poiss_model('dif', '$A_\mathrm{dif}$', [-5,2], True)
# n.add_poiss_model('iso', '$A_\mathrm{iso}$', [-5,2], True)
# n.add_poiss_model('psc', '$A_\mathrm{psc}$', [-5,2], True)
# n.add_poiss_model('bub', '$A_\mathrm{bub}$', [-5,2], True)
# n.add_poiss_model('nfw_dm', '$A_\mathrm{nfw_{dm}}$', [-5,2], True)

# n.add_non_poiss_model('iso_ps',
# 					  ['$\log_{10}(A_\mathrm{iso}^\mathrm{ps})$','$n_1^\mathrm{iso}$','$n_2^\mathrm{iso}$','$n_3^\mathrm{iso}$','$S_b^{(1),\mathrm{iso}}$','$S_b^{(2),\mathrm{iso}}$'],
# 					  [[-10,5],[2.05,15.],[-3.95,2.95],[-10,1.95],[0.5,3.5],[-1.75,0.5]],
# 					  [True,False,False,False,True,True])

# n.add_non_poiss_model('dsk_ps',
# 					  ['$\log_{10}(A_\mathrm{dsk}^\mathrm{ps})$','$n_1^\mathrm{dsk}$','$n_2^\mathrm{dsk}$','$n_3^\mathrm{dsk}$','$S_b^{(1),\mathrm{dsk}}$','$S_b^{(2),\mathrm{dsk}}$'],
# 					  [[-10,5],[2.05,15.],[-3.95,2.95],[-10,1.95],[0.5,3.5],[-1.75,0.5]],
# 					  [True,False,False,False,True,True])

if model_GCE:
	n.add_non_poiss_model('nfw_ps',
						  ['$\log_{10}(A_\mathrm{nfw}^\mathrm{ps})$','$n_1^\mathrm{nfw}$','$n_2^\mathrm{nfw}$','$n_3^\mathrm{nfw}$','$S_b^{(1),\mathrm{nfw}}$','$S_b^{(2),\mathrm{nfw}}$'],
						  [[-10,5],[2.05,15.],[-3.95,2.95],[-10,1.95],[0.5,3.5],[-1.75,0.5]],
						  [True,False,False,False,True,True])

if model_PSF:
	if psf_king:
		print("Using King PSF function")
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
		print("Using gaussian PSF")
		pc_inst = pc.PSFCorrection(psf_sigma_deg=0.1812)
		f_ary, df_rho_div_f_ary = pc_inst.f_ary, pc_inst.df_rho_div_f_ary

	n.configure_for_scan(f_ary, df_rho_div_f_ary, nexp=1)
else:
	n.configure_for_scan(nexp=1)

n.perform_scan(nlive=300, 
pymultinest_options = {'importance_nested_sampling': False,
					   'resume': False, 'verbose': True,
					   'sampling_efficiency': 'model',
					   'init_MPI': False, 'evidence_tolerance': 0.5,
					   'const_efficiency_mode': False})
