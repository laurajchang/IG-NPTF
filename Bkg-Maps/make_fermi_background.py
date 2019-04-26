###############################################################################
# make_fermi_background.py
###############################################################################
# HISTORY:
#   2017-11-22 - Written - Laura Chang (Princeton)
###############################################################################

import os, sys
import argparse
import numpy as np
import healpy as hp
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg')

# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
# from NPTFit import dnds_analysis # module for analysing the output

# # Feynman dirs
# work_dir = '/group/hepheno/smsharma/Fermi-LSS/'
# psf_dir='/mnt/hepheno/CTBCORE/psf_data/'
# maps_dir='/mnt/hepheno/CTBCORE/'
# fermi_data_dir='/mnt/hepheno/FermiData/'
# data_folder='/tigress/Fermi-LSS/AdditionalData/'

# Parse input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--emin',action='store', dest='emin', default=0,type=int)
parser.add_argument('--emax',action='store', dest='emax', default=2,type=int)
parser.add_argument('--MC_num',action='store', dest='MC_num', default=1,type=int)
parser.add_argument('--hemisphere',action='store', dest='hemisphere', default="",type=str)
parser.add_argument('--hemi_b',action='store', dest='hemi_b', default="",type=str)
parser.add_argument('--hemi_l',action='store', dest='hemi_l', default="",type=str)
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
parser.add_argument("--mask_eighths", action="store", dest="mask_eighths", default=0,type=int)
parser.add_argument("--mask_eighths_ind", action="store", dest="mask_eighths_ind", default=0,type=int)
parser.add_argument("--mask_sixteenths", action="store", dest="mask_sixteenths", default=0,type=int)
parser.add_argument("--mask_sixteenths_ind", action="store", dest="mask_sixteenths_ind", default=0,type=int)
parser.add_argument('--temp_count',action='store', dest='temp_count', default=1,type=int)
parser.add_argument('--diff',action='store', dest='diff', default='p6',type=str)
parser.add_argument('--save_dir',action='store', dest='save_dir', default='/tigress/ljchang/',type=str)

results = parser.parse_args()
emin = results.emin
emax = results.emax
MC_num = results.MC_num
hemisphere = results.hemisphere
hemi_b = results.hemi_b
hemi_l = results.hemi_l
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
mask_eighths = results.mask_eighths
mask_eighths_ind = results.mask_eighths_ind
mask_sixteenths = results.mask_sixteenths
mask_sixteenths_ind = results.mask_sixteenths_ind
temp_count = results.temp_count
diff = results.diff
save_dir = results.save_dir

mask_options_dict={"band_mask":mask_band,"band_mask_range":mask_bandval,"mask_ring":mask_ring,"inner":mask_innerval,"outer":mask_outerval,
"b_mask":mask_b,"b_deg_min":mask_bmin,"b_deg_max":mask_bmax,"l_mask":mask_l,"l_deg_min":mask_lmin,"l_deg_max":mask_lmax}

# mask_tag = '_band_' + str(mask_bandval) + '_ring_' + str(mask_outerval)+ '_' + hemisphere
# mask_tag = '_band_' + str(mask_bandval) + '_ring_' + str(mask_outerval)+ hemi_b + hemi_l

# Make save dir if it doesn't exist
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

# Global settings
nside=128
eventclass=5 # 2 (Source) or 5 (UltracleanVeto)
eventtype=3 # 0 (all), 3 (bestpsf) or 5 (top3 quartiles)
# diff = 'p6' # 'p6', 'p7', 'p8'
print "Using diffuse model ",diff

###################################################
# Loop over bins and masks and get best-fit norms #
###################################################

data_file_path = '/tigress/smsharma/Fermi-SmoothGalHalo/DataFiles/Bkg-Maps/fermi_data/eventclass'+str(eventclass)+'_eventtype'+str(eventtype)+'/'

if mask_eighths:
	print "Masking eighths!"
	# masks = np.load('/tigress/smsharma/Fermi-SmoothGalHalo/DataFiles/masks_eighths_b_20_r_50.npy')
	masks = np.load('/tigress/smsharma/Fermi-SmoothGalHalo/DataFiles/masks_eighths_b_15_r_50.npy')	
	mask_tags = ["N1","N2","N3","N4","S1","S2","S3","S4"]

	analysis_mask_base = masks[mask_eighths_ind]
	mask_tag = '_band_' + str(mask_bandval) + '_ring_' + str(mask_outerval) + '_' + mask_tags[mask_eighths_ind]
	temp_count = mask_tags[mask_eighths_ind]

elif mask_sixteenths:
	print "Masking sixteenths!"
	masks = np.load('/tigress/smsharma/Fermi-SmoothGalHalo/DataFiles/masks_sixteenths_b_20_r_50.npy')
	mask_tags = ["N1_1","N1_2","N2_1","N2_2","N3_1","N3_2","N4_1","N4_2","S1_1","S1_2","S2_1","S2_2","S3_1","S3_2","S4_1","S4_2"]	

	analysis_mask_base = masks[mask_sixteenths_ind]
	mask_tag = '_band_' + str(mask_bandval) + '_ring_' + str(mask_outerval) + '_' + mask_tags[mask_sixteenths_ind]
	temp_count = mask_tags[mask_sixteenths_ind]

else:
	analysis_mask_base = cm.make_mask_total(**mask_options_dict)
	if mask_innerval != 0:
		mask_tag = '_band_' + str(mask_bandval) + '_ring_' + str(mask_innerval) + '_' + str(mask_outerval)+ hemi_b + hemi_l	
	else:
		mask_tag = '_band_' + str(mask_bandval) + '_ring_' + str(mask_outerval)+ hemi_b + hemi_l

print "Total number of pixels in mask base is ", np.sum(~analysis_mask_base)
	
# ABC_path = '/tigress/smsharma/Fermi-SmoothGalHalo/DataFiles/Bkg-Maps/diffuse_models_ABC/'
# brem_file = np.load(ABC_path+'model'+diff+'_brem.npy')
# pi0_file = np.load(ABC_path+'model'+diff+'_pi0.npy')
# ics_file = np.load(ABC_path+'model'+diff+'_ics.npy')

for iebin, ebin in enumerate(range(emin, emax+1)):

	print 'At bin ', ebin + 1, ', of', str(emax-emin+1)

	# Get templates and maps from plugin

	fermi_data = np.load(data_file_path+'fermi_data_ebin_'+str(ebin)+'.npy')
	fermi_exposure = np.load(data_file_path+'fermi_exposure_ebin_'+str(ebin)+'.npy')
	# brem = hp.ud_grade(brem_file[ebin],nside,power=-2)
	# pi0 = hp.ud_grade(pi0_file[ebin],nside,power=-2)
	# ics = hp.ud_grade(ics_file[ebin],nside,power=-2)
	print "Using diffuse template ", diff
	dif = np.load(data_file_path+'dif_'+diff+'_ebin_'+str(ebin)+'.npy')
	iso = np.load(data_file_path+'iso_ebin_'+str(ebin)+'.npy')
	# psc = np.load(data_file_path+'psc_ebin_'+str(ebin)+'.npy')
	psc = np.load('/tigress/smsharma/Fermi-SmoothGalHalo/DataFiles/PS-Maps/ps_map_et3.npy')[ebin]
	bub = np.load(data_file_path+'bub_ebin_'+str(ebin)+'.npy')

	###################################################################
	# Set up LL with NPTFit, then use scipy.minimize to find best-fit #
 	###################################################################

 	n = nptfit.NPTF(tag='b_'+str(ebin)+'_'+str(temp_count))
 	n.load_data(fermi_data, fermi_exposure)

	# Mask used in analysis

	ps_mask = np.load(data_file_path+'psc_mask_0.95_ebin_'+str(ebin)+'.npy')
	analysis_mask = analysis_mask_base + ps_mask

	analysis_mask = analysis_mask > 0 
	
	n.load_mask(analysis_mask)

	n.add_template(dif, diff)
	# n.add_template(brem, 'brem_model'+diff)
	# n.add_template(pi0, 'pi0_model'+diff)
	# n.add_template(ics, 'ics_model'+diff)
	n.add_template(iso, 'iso')
	n.add_template(psc, 'psc')
	n.add_template(bub, 'bub')

	n.add_poiss_model(diff, '$A_\mathrm{dif}$', [0,10], False)
	# n.add_poiss_model('brem_model'+diff, '$A_\mathrm{brem}$', [0,10], False)
	# n.add_poiss_model('pi0_model'+diff, '$A_\mathrm{pi0}$', [0,10], False)
	# n.add_poiss_model('ics_model'+diff, '$A_\mathrm{ics}$', [0,10], False)
	n.add_poiss_model('iso', '$A_\mathrm{iso}$', [0,20], False)
	n.add_poiss_model('psc', '$A_\mathrm{psc}$', [0,10], False)
	n.add_poiss_model('bub', '$A_\mathrm{bub}$', [0,10], False)

	n.configure_for_scan()

	scpy_min = minimize(lambda x: -n.ll(x), x0=[1e-10 for i in range(len(n.poiss_models))], bounds=[[0.,None] for i in range(len(n.poiss_models))], options={'disp':False,'ftol':1e-12}, method='L-BFGS-B')
	best_fit_params = scpy_min['x']

	# print best_fit_params

	dif_temp = best_fit_params[0]*dif
	iso_temp = best_fit_params[1]*iso
	psc_temp = best_fit_params[2]*psc
	bub_temp = best_fit_params[3]*bub

	best_fit_bkg = dif_temp+iso_temp+psc_temp+bub_temp

	# brem_temp = best_fit_params[0]*brem
	# pi0_temp = best_fit_params[1]*pi0
	# ics_temp = best_fit_params[2]*ics
	# iso_temp = best_fit_params[3]*iso
	# psc_temp = best_fit_params[4]*psc
	# bub_temp = best_fit_params[5]*bub

	# best_fit_bkg = brem_temp+pi0_temp+ics_temp+iso_temp+psc_temp+bub_temp

	print best_fit_params, np.sum(~analysis_mask*fermi_data), np.sum(~analysis_mask*best_fit_bkg)

	###########################
	# Save best-fit templates #
 	###########################

 	np.save(save_dir + '/dif_ebin_' + str(ebin) + mask_tag + '.npy', dif_temp)
 	# np.save(save_dir + '/brem_model' + diff + '_ebin_' + str(ebin) + mask_tag + '.npy', brem_temp)
 	# np.save(save_dir + '/pi0_model' + diff + '_ebin_' + str(ebin) + mask_tag + '.npy', pi0_temp)
 	# np.save(save_dir + '/ics_model' + diff + '_ebin_' + str(ebin) + mask_tag + '.npy', brem_temp)
 	# np.save(save_dir + '/brem_pi0_combined_model' + diff + '_ebin_' + str(ebin) + mask_tag + '.npy', brem_temp+pi0_temp)
 	np.save(save_dir + '/iso_ebin_' + str(ebin) + mask_tag + '.npy', iso_temp)
 	np.save(save_dir + '/psc_ebin_' + str(ebin) + mask_tag + '.npy', psc_temp)
 	np.save(save_dir + '/bub_ebin_' + str(ebin) + mask_tag + '.npy', bub_temp)
 	np.save(save_dir + '/best_fit_norms_ebin_' + str(ebin) + mask_tag + '.npy', best_fit_params)
 	# np.save(save_dir + '/best_fit_bkg_ebin_' + str(ebin) + mask_tag + '.npy', best_fit_bkg)
