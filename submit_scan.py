import os, sys
import numpy as np

batch1='''#!/bin/bash
#SBATCH -N 2   # node count
#SBATCH -n 20
#SBATCH -t 12:00:00
#SBATCH --mem=50GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=ljchang@princeton.edu

cd /tigress/ljchang/NPTF-IG-Check/
		'''


############################################################
# Running NPTF scan on signal injection MCs, with king PSF #
############################################################

work_dir = "/tigress/ljchang/NPTF-IG-Check/"
save_dir = "/chains/"
# Make save dir if it doesn't exist
if not os.path.exists(work_dir+save_dir):
	os.makedirs(work_dir+save_dir)

GCE_options = [0,1]
GCE_tags = ["","_with_NFWPS"]

# file_tag_types = ["NFW_PS_1bk_2048_dif","NFW_PS_below1ph_2048_dif","NFW_PS_below1ph_flat_2048_dif"]
# out_tag_names = ["PS_1.00/PS_1bk/PS_dif/","PS_1.00/PS_below1ph/PS_dif/","PS_1.00/PS_below1ph_flat/PS_dif/"]

# file_tag_types = ["DM_sim_40GeV_1em26"]
# out_tag_names = ["DM_1.00/DM_only/"]

# file_tag_types = ["DM_sim_40GeV_1em26_dif"]
# out_tag_names = ["DM_1.00/DM_dif/"]

# file_tag_types = ["NFW_PS_below1ph_flat_2048"]
# out_tag_names = ["simple_scans/no_PSF/PS_only/"]

file_tag_types = ["NFW_PS_below1ph_flat_2048_dif"]
out_tag_names = ["simple_scans/test"]

# file_tag_types = ["NFW_PS_1bk_2048","NFW_PS_below1ph_2048","NFW_PS_below1ph_flat_2048"]
# out_tag_names = ["PS_1.00/PS_1bk/PS_only/","PS_1.00/PS_below1ph/PS_only/","PS_1.00/PS_below1ph_flat/PS_only/"]

# file_tag_types = ["NFW_PS_1bk_2048_0.25GCEflux_nfwDM_0.75GCEflux_dif_iso",\
# 				"NFW_PS_1bk_2048_0.5GCEflux_nfwDM_0.5GCEflux_dif_iso",\
# 				"NFW_PS_1bk_2048_0.75GCEflux_nfwDM_0.25GCEflux_dif_iso"]

# out_tag_names = ["PS_0.25_DM_0.75/PS_1bk/",\
# 				"PS_0.5_DM_0.5/PS_1bk/",\
# 				"PS_0.75_DM_0.25/PS_1bk/"]

# file_tag_types = ["NFW_PS_below1ph_2048_0.25GCEflux_nfwDM_0.75GCEflux_dif_iso",\
# 				"NFW_PS_below1ph_2048_0.5GCEflux_nfwDM_0.5GCEflux_dif_iso",\
# 				"NFW_PS_below1ph_2048_0.75GCEflux_nfwDM_0.25GCEflux_dif_iso"]

# out_tag_names = ["PS_0.25_DM_0.75/PS_below1ph/",\
# 				"PS_0.5_DM_0.5/PS_below1ph/",\
# 				"PS_0.75_DM_0.25/PS_below1ph/"]

# for iGCE in [0,1]:

# b_outer_ary = [[2,15],[2,30],[5,15],[5,30]]
# b_outer_ary = [[2,15]]

# for iGCE in [1]:
# 	model_GCE = GCE_options[iGCE]
# 	for ifile in range(len(file_tag_types)):
# 	# for ifile in [1]:
# 		file_tag_type = file_tag_types[ifile]
# 		run_tag_base = out_tag_names[ifile]

# 		# data_dir = "/tigress/ljchang/NPTF-IG-Check/data/DMsim/"
# 		# data_file_path = data_dir+file_tag_type+".npy"
# 		for iROI in range(len(b_outer_ary)):
# 			bval, outerval = b_outer_ary[iROI]
# 			innerval = 0

# 			if innerval == 0:
# 				data_dir = "/tigress/ljchang/NPTF-IG-Check/data/simPS/MCs/with_bkg_components/flat_exposure/king_smoothed/"
# 				data_file_path = data_dir+"varyROI/b_"+str(bval)+"_r_"+str(outerval)+"/"+file_tag_type+".npy"

# 				run_tag = run_tag_base+"b_"+str(bval)+"_r_"+str(outerval)
# 			else:
# 				data_dir = "/tigress/ljchang/NPTF-IG-Check/data/simPS/MCs/with_bkg_components/flat_exposure/king_smoothed/"
# 				data_file_path = data_dir+"varyROI/b_"+str(bval)+"_r_"+str(innerval)+"_"+str(outerval)+"/"+file_tag_type+".npy"

# 				run_tag = run_tag_base+"b_"+str(bval)+"_r_"+str(innerval)+"_"+str(outerval)				
# 			# data_file_path = "/tigress/ljchang/NPTF-IG-Check/data/DMsim/DM_sim_40GeV_8p5em27_dif_iso.npy"
# 			# run_tag = run_tag_base+"dif_iso_nfw_log2"

# 			# run_tag = "PStests/PS_nfwDM_dif_iso/all_p8"

# 			b = bval
# 			inner = innerval
# 			outer = outerval

# 			batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
# 						'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+'mask_innerval='+str(inner)+'\n'+ \
# 						'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+'psf_king='+str(1)+'\n'+\
# 						'data_file_path='+data_file_path+'\n'+'work_dir='+work_dir+'\n'
# 			batch3 = 'mpirun -np 20 python np_scan_simple.py --mask_band $mask_band --mask_ring $mask_ring '
# 			# batch3 = 'mpirun -np 20 python np_scan_1bk.py --mask_band $mask_band --mask_ring $mask_ring '
# 			batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --mask_innerval $mask_innerval --model_GCE $model_GCE '
# 			batch5 = '--run_tag $run_tag --psf_king $psf_king --data_file_path $data_file_path --work_dir $work_dir\n'
# 			batchn = batch1+batch2+batch3+batch4+batch5
# 			# fname = './batch/scan_'+file_tag_type+'_'+str(MC)+GCE_tags[iGCE]+'_king.batch'
# 			fname = './batch/'+run_tag+'.batch'
# 			f=open(fname, "w")
# 			f.write(batchn)
# 			f.close()
# 			os.system("sbatch "+fname);

# b_outer_ary = [[2,30]]

# for iGCE in [1]:
# 	model_GCE = GCE_options[iGCE]
# 	for ifile in range(len(file_tag_types)):
# 	# for ifile in [1]:
# 		file_tag_type = file_tag_types[ifile]
# 		run_tag_base = out_tag_names[ifile]

# 		# data_dir = "/tigress/ljchang/NPTF-IG-Check/data/DMsim/"
# 		# data_file_path = data_dir+file_tag_type+".npy"
# 		for iROI in range(len(b_outer_ary)):
# 			bval, outerval = b_outer_ary[iROI]

# 			data_dir = "/tigress/ljchang/NPTF-IG-Check/data/simPS/MCs/with_bkg_components/flat_exposure/king_smoothed/"
# 			# data_file_path = data_dir+"varyROI/b_"+str(bval)+"_r_"+str(outerval)+"/"+file_tag_type+".npy"
# 			data_file_path = data_dir+file_tag_type+".npy"
# 			run_tag = run_tag_base

# 			# run_tag = run_tag_base+"b_"+str(bval)+"_r_"+str(outerval)

# 			# data_file_path = "/tigress/ljchang/NPTF-IG-Check/data/DMsim/DM_sim_40GeV_8p5em27_dif_iso.npy"
# 			# run_tag = run_tag_base+"dif_iso_nfw_log2"

# 			# run_tag = "PStests/PS_nfwDM_dif_iso/all_p8"

# 			b = bval
# 			outer = outerval

# 			batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
# 						'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 						'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+'psf_king='+str(1)+'\n'+\
# 						'data_file_path='+data_file_path+'\n'+'work_dir='+work_dir+'\n'
# 			batch3 = 'mpirun -np 20 python np_scan_simple.py --mask_band $mask_band --mask_ring $mask_ring '
# 			# batch3 = 'mpirun -np 20 python np_scan_1bk.py --mask_band $mask_band --mask_ring $mask_ring '
# 			batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
# 			batch5 = '--run_tag $run_tag --psf_king $psf_king --data_file_path $data_file_path --work_dir $work_dir\n'
# 			batchn = batch1+batch2+batch3+batch4+batch5
# 			# fname = './batch/scan_'+file_tag_type+'_'+str(MC)+GCE_tags[iGCE]+'_king.batch'
# 			fname = './batch/'+run_tag+'.batch'
# 			f=open(fname, "w")
# 			f.write(batchn)
# 			f.close()
# 			os.system("sbatch "+fname);

######################################################
# Running NPTF scan on background MCs, with king PSF #
######################################################

# work_dir = "/tigress/ljchang/NPTF-IG-Check/"
# save_dir = "/chains/"
# # Make save dir if it doesn't exist
# if not os.path.exists(work_dir+save_dir):
# 	os.makedirs(work_dir+save_dir)

# GCE_options = [0,1]
# GCE_tags = ["","_with_NFWPS"]

# file_tag_types = ["NFW_PS_1bk","NFW_PS_below1ph","NFW_PS_below1ph_flat","NFW_PS_below1ph_rising"]
# out_tag_names = ["PS_1bk","PS_below1ph","PS_below1ph_flat","PS_below1ph_rising"]

# # file_tag_types = ["NFW_PS_below1ph_2048_bub"]
# # out_tag_names = ["PS_below1ph_2048"]

# # for iGCE in [0,1]:
# for iGCE in [0]:
# 	model_GCE = GCE_options[iGCE]
# 	data_file_path = "/tigress/ljchang/NPTF-IG-Check/data/dif_iso_MC.npy"
# 	# run_tag = "PS_only/scan_"+out_tag_names[ifile]+GCE_tags[iGCE]
# 	run_tag = "dif_iso_MC_log"

# 	b = 2
# 	outer = 30

# 	batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
# 				'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 				'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+'psf_king='+str(1)+'\n'+\
# 				'data_file_path='+data_file_path+'\n'+'work_dir='+work_dir+'\n'
# 	batch3 = 'mpirun -np 20 python np_scan_mask0p8deg_copy.py --mask_band $mask_band --mask_ring $mask_ring '
# 	batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
# 	batch5 = '--run_tag $run_tag --psf_king $psf_king --data_file_path $data_file_path --work_dir $work_dir\n'
# 	batchn = batch1+batch2+batch3+batch4+batch5
# 	# fname = './batch/scan_'+file_tag_type+'_'+str(MC)+GCE_tags[iGCE]+'_king.batch'
# 	fname = './batch/'+run_tag+'_king.batch'
# 	f=open(fname, "w")
# 	f.write(batchn)
# 	f.close()
# 	os.system("sbatch "+fname);

########################################################
# work_dir = "/tigress/ljchang/NPTF-IG-Check/"
# save_dir = "/chains/"
# # Make save dir if it doesn't exist
# if not os.path.exists(work_dir+save_dir):
# 	os.makedirs(work_dir+save_dir)

# GCE_options = [0,1]
# GCE_tags = ["","_with_NFWPS"]

# # for iGCE in [0,1]:
# for iGCE in [1]:
# 	model_GCE = GCE_options[iGCE]
# 	data_file_path = "/tigress/ljchang/NPTF-IG-Check/data/DM_sim_40GeV_1p3em26.npy"
# 	# run_tag = "PS_only/scan_"+out_tag_names[ifile]+GCE_tags[iGCE]
# 	run_tag = "DMtests/dif_bub_nfw_nfwPS"

# 	b = 2
# 	outer = 30

# 	batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
# 				'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 				'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+'psf_king='+str(1)+'\n'+\
# 				'data_file_path='+data_file_path+'\n'+'work_dir='+work_dir+'\n'
# 	batch3 = 'mpirun -np 20 python np_scan_mask0p8deg.py --mask_band $mask_band --mask_ring $mask_ring '
# 	batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
# 	batch5 = '--run_tag $run_tag --psf_king $psf_king --data_file_path $data_file_path --work_dir $work_dir\n'
# 	batchn = batch1+batch2+batch3+batch4+batch5
# 	# fname = './batch/scan_'+file_tag_type+'_'+str(MC)+GCE_tags[iGCE]+'_king.batch'
# 	fname = './batch/'+run_tag+'_king.batch'
# 	f=open(fname, "w")
# 	f.write(batchn)
# 	f.close()
# 	os.system("sbatch "+fname);

###########################################
# Running NPTF scan on data with King PSF #
###########################################

work_dir = "/tigress/ljchang/NPTF-IG-Check/"
save_dir = "/chains/"
# Make save dir if it doesn't exist
if not os.path.exists(work_dir+save_dir):
	os.makedirs(work_dir+save_dir)

GCE_options = [0,1]
GCE_tags = ["","_with_NFWPS"]

# for iGCE in [0,1]:
for iGCE in [0]:
	model_GCE = GCE_options[iGCE]
	run_tag = "mpmath/scan_data_poiss_lin"+GCE_tags[iGCE]
	# run_tag = "scan_data_lin"

	b = 2
	outer = 30

	batch2 = 'scan_data='+str(1)+'\n'+'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
				'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
				'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+\
				'psf_king='+str(1)+'\n'+'work_dir='+work_dir+'\n'
	batch3 = 'mpiexec -np 20 python np_scan_simple.py --scan_data $scan_data --mask_band $mask_band --mask_ring $mask_ring '
	# batch3 = 'mpiexec -np 20 python np_scan_no_isops.py --scan_data $scan_data --mask_band $mask_band --mask_ring $mask_ring '
	batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
	batch5 = '--run_tag $run_tag --psf_king $psf_king --work_dir $work_dir\n'
	batchn = batch1+batch2+batch3+batch4+batch5
	fname = './batch/'+run_tag+'.batch'

	# fname = './batch/make_bkg_'+str(emin)+'_'+str(emax)+'_model'+diff+'.batch'
	f=open(fname, "w")
	f.write(batchn)
	f.close()
	os.system("sbatch "+fname);

#########################
# Data signal injection #
#########################

# work_dir = "/tigress/ljchang/NPTF-IG-Check/"
# save_dir = "/chains/"
# # Make save dir if it doesn't exist
# if not os.path.exists(work_dir+save_dir):
# 	os.makedirs(work_dir+save_dir)

# GCE_options = [0,1]
# GCE_tags = ["","_with_NFWPS"]

# # for iGCE in [0,1]:
# xsec_tags = ["8p5em27","4p1em26","8p1em26"]

# iGCE = 1
# for xsec_tag in xsec_tags:
# 	model_GCE = GCE_options[iGCE]
# 	data_dir = "/tigress/ljchang/NPTF-IG-Check/data/"
# 	data_file_path = data_dir+"data_sig_inj_40GeV_"+xsec_tag+".npy"
# 	# run_tag = "data_runs/inj_sig_PSbelow1ph_rising_100"+GCE_tags[iGCE]
# 	run_tag = "data_inj_sig_"+xsec_tag+"_lin"

# 	b = 2
# 	outer = 30

# 	batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
# 				'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 				'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+'psf_king='+str(1)+'\n'+\
# 				'data_file_path='+data_file_path+'\n'+'work_dir='+work_dir+'\n'
# 	batch3 = 'mpirun -np 20 python np_scan_mask0p8deg_copy.py --mask_band $mask_band --mask_ring $mask_ring '
# 	# batch3 = 'mpirun -np 20 python np_scan_no_isops.py --mask_band $mask_band --mask_ring $mask_ring '
# 	batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
# 	batch5 = '--run_tag $run_tag --psf_king $psf_king --data_file_path $data_file_path --work_dir $work_dir\n'
# 	batchn = batch1+batch2+batch3+batch4+batch5
# 	# fname = './batch/scan_'+file_tag_type+'_'+str(MC)+GCE_tags[iGCE]+'_king.batch'
# 	fname = './batch/'+run_tag+'.batch'
# 	f=open(fname, "w")
# 	f.write(batchn)
# 	f.close()
# 	os.system("sbatch "+fname);