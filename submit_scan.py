import os, sys
import numpy as np

batch1='''#!/bin/bash
##SBATCH -N 1   # node count
#SBATCH -n 20
#SBATCH -t 12:00:00
#SBATCH --mem=50GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=ljchang@princeton.edu

cd /tigress/ljchang/NPTF-IG-Check/
		'''

#############################################
# Running NPTF scan on signal injection MCs #
#############################################

# work_dir = "/tigress/ljchang/NPTF-IG-Check/"
# save_dir = "/chains/"
# # Make save dir if it doesn't exist
# if not os.path.exists(work_dir+save_dir):
# 	os.makedirs(work_dir+save_dir)

# GCE_options = [0,1]
# GCE_tags = ["","_with_GCE_ps"]

# for iGCE in [0,1]:
# 	model_GCE = GCE_options[iGCE]
# 	for xsec_ind in range(4):
# 		for MC in [0]:
# 			data_dir = "/tigress/ljchang/NPTF-IG-Check/data/injected_signal_MC/100GeV/"
# 			data_file_path = data_dir+"xbin_"+str(xsec_ind)+"_"+str(MC)+".npy"
# 			run_tag = "xbin_"+str(xsec_ind)+"_"+str(MC)+GCE_tags[iGCE]

# 			b = 2
# 			outer = 30

# 			batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
# 						'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 						'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+\
# 						'data_file_path='+data_file_path+'\n'+'work_dir='+work_dir+'\n'
# 			batch3 = 'mpirun -np 20 python np_scan.py --mask_band $mask_band --mask_ring $mask_ring '
# 			batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
# 			batch5 = '--run_tag $run_tag --data_file_path $data_file_path --work_dir $work_dir\n'
# 			batchn = batch1+batch2+batch3+batch4+batch5
# 			fname = './batch/scan_xsec_'+str(xsec_ind)+'_'+str(MC)+GCE_tags[iGCE]+'.batch'
# 			# fname = './batch/make_bkg_'+str(emin)+'_'+str(emax)+'_model'+diff+'.batch'
# 			f=open(fname, "w")
# 			f.write(batchn)
# 			f.close()
# 			os.system("sbatch "+fname);

############################################################
# Running NPTF scan on signal injection MCs, with king PSF #
############################################################

work_dir = "/tigress/ljchang/NPTF-IG-Check/"
save_dir = "/chains/"
# Make save dir if it doesn't exist
if not os.path.exists(work_dir+save_dir):
	os.makedirs(work_dir+save_dir)

GCE_options = [0,1]
GCE_tags = ["","_with_GCE_ps"]

# file_tag_types1 = ["PS_1bk_100","PS_1bk_smoothDM_40GeV_100_100","PS_1bk_smoothDM_40GeV_50_50"]
# out_tag_names1 = ["PS_1bk_100","PS_1bk_DM_100_100","PS_1bk_DM_50_50"]

# file_tag_types2 = ["PS_below1ph_100","PS_below1ph_smoothDM_40GeV_100_100","PS_below1ph_smoothDM_40GeV_50_50"]
# out_tag_names2 = ["below1phPS_100","below1phPS_DM_100_100","below1phPS_DM_50_50"]

# file_tag_types = file_tag_types1+file_tag_types2
# out_tag_names = out_tag_names1+out_tag_names2

file_tag_types = ["PS_below1ph_smoothDM_40GeV_100_100"]
out_tag_names = ["below1phPS_DM_100_100"]

for iGCE in [0,1]:
# for iGCE in [0]:
	model_GCE = GCE_options[iGCE]
	for ifile in range(len(file_tag_types)):
		file_tag_type = file_tag_types[ifile]
		for MC in [0]:
			data_dir = "/tigress/ljchang/NPTF-IG-Check/data/GCE_MC/king_smoothed/"
			data_file_path = data_dir+file_tag_type+"_"+str(MC)+".npy"
			run_tag = "king/mask_0p8/scan_"+out_tag_names[ifile]+GCE_tags[iGCE]

			b = 2
			outer = 30

			batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
						'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
						'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+'psf_king='+str(1)+'\n'+\
						'data_file_path='+data_file_path+'\n'+'work_dir='+work_dir+'\n'
			batch3 = 'mpirun -np 20 python np_scan.py --mask_band $mask_band --mask_ring $mask_ring '
			batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
			batch5 = '--run_tag $run_tag --psf_king $psf_king --data_file_path $data_file_path --work_dir $work_dir\n'
			batchn = batch1+batch2+batch3+batch4+batch5
			# fname = './batch/scan_'+file_tag_type+'_'+str(MC)+GCE_tags[iGCE]+'_king.batch'
			fname = './batch/'+run_tag+'_king.batch'
			f=open(fname, "w")
			f.write(batchn)
			f.close()
			os.system("sbatch "+fname);

########################################################

# work_dir = "/tigress/ljchang/NPTF-IG-Check/"
# save_dir = "/chains/"
# # Make save dir if it doesn't exist
# if not os.path.exists(work_dir+save_dir):
# 	os.makedirs(work_dir+save_dir)

# GCE_options = [0,1]
# GCE_tags = ["","_with_GCE_ps"]

# # for iGCE in [0,1]:
# for iGCE in [1]:
# 	model_GCE = GCE_options[iGCE]
# 	for file_tag_type in ["bubPS_smoothDM_40GeV_0p7em26_50_50"]:
# 		for MC in [0]:
# 			data_dir = "/tigress/ljchang/NPTF-IG-Check/data/GCE_MC/king_smoothed/"
# 			data_file_path = data_dir+file_tag_type+"_"+str(MC)+".npy"
# 			# run_tag = "king/scan_"+file_tag_type+"_"+str(MC)+GCE_tags[iGCE]
# 			run_tag = "king/bubPS_DM/scan_50_50"

# 			b = 2
# 			outer = 30

# 			batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
# 						'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 						'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+'psf_king='+str(1)+'\n'+\
# 						'data_file_path='+data_file_path+'\n'+'work_dir='+work_dir+'\n'
# 			batch3 = 'mpirun -np 20 python np_scan.py --mask_band $mask_band --mask_ring $mask_ring '
# 			batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
# 			batch5 = '--run_tag $run_tag --psf_king $psf_king --data_file_path $data_file_path --work_dir $work_dir\n'
# 			batchn = batch1+batch2+batch3+batch4+batch5
# 			# fname = './batch/scan_'+file_tag_type+'_'+str(MC)+GCE_tags[iGCE]+'_king.batch'
# 			fname = './batch/'+run_tag+'_king.batch'
# 			f=open(fname, "w")
# 			f.write(batchn)
# 			f.close()
# 			os.system("sbatch "+fname);

########################################################

# work_dir = "/tigress/ljchang/NPTF-IG-Check/"
# save_dir = "/chains/"
# # Make save dir if it doesn't exist
# if not os.path.exists(work_dir+save_dir):
# 	os.makedirs(work_dir+save_dir)

# GCE_options = [0,1]
# GCE_tags = ["","_with_GCE_ps"]

# for iGCE in [0,1]:
# # for iGCE in [1]:
# 	model_GCE = GCE_options[iGCE]
# 	for file_tag_type in ["twice_smoothDM_40GeV_1p3em26"]:
# 		data_dir = "/tigress/ljchang/NPTF-IG-Check/data/GCE_MC/king_smoothed/"
# 		data_file_path = data_dir+file_tag_type+".npy"
# 		run_tag = "king/scan_twice_smoothDM"+GCE_tags[iGCE]

# 		b = 2
# 		outer = 30

# 		batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
# 					'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 					'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+'psf_king='+str(1)+'\n'+\
# 					'data_file_path='+data_file_path+'\n'+'work_dir='+work_dir+'\n'
# 		batch3 = 'mpirun -np 20 python np_scan.py --mask_band $mask_band --mask_ring $mask_ring '
# 		batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
# 		batch5 = '--run_tag $run_tag --psf_king $psf_king --data_file_path $data_file_path --work_dir $work_dir\n'
# 		batchn = batch1+batch2+batch3+batch4+batch5
# 		# fname = './batch/scan_'+file_tag_type+'_'+str(MC)+GCE_tags[iGCE]+'_king.batch'
# 		fname = './batch/'+run_tag+'_king.batch'
# 		f=open(fname, "w")
# 		f.write(batchn)
# 		f.close()
# 		os.system("sbatch "+fname);

#############################
# Running NPTF scan on data #
#############################

# work_dir = "/tigress/ljchang/NPTF-IG-Check/"
# save_dir = "/chains/"
# # Make save dir if it doesn't exist
# if not os.path.exists(work_dir+save_dir):
# 	os.makedirs(work_dir+save_dir)

# GCE_options = [0,1]
# GCE_tags = ["","_with_GCE_ps"]

# # for iGCE in [0,1]:
# for iGCE in [1]:
# 	model_GCE = GCE_options[iGCE]
# 	run_tag = "scan_data_unmasked_gamma1p2_2"+GCE_tags[iGCE]

# 	b = 2
# 	outer = 30

# 	batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
# 				'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 				'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+\
# 				'work_dir='+work_dir+'\n'
# 	batch3 = 'mpiexec -np 20 python np_scan_unmasked.py --mask_band $mask_band --mask_ring $mask_ring '
# 	batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
# 	batch5 = '--run_tag $run_tag --work_dir $work_dir\n'
# 	batchn = batch1+batch2+batch3+batch4+batch5
# 	fname = './batch/'+run_tag+'.batch'
# 	# fname = './batch/make_bkg_'+str(emin)+'_'+str(emax)+'_model'+diff+'.batch'
# 	f=open(fname, "w")
# 	f.write(batchn)
# 	f.close()
# 	os.system("sbatch "+fname);


###################################
# Running NPTF scan with King PSF #
###################################

# work_dir = "/tigress/ljchang/NPTF-IG-Check/"
# save_dir = "/chains/"
# # Make save dir if it doesn't exist
# if not os.path.exists(work_dir+save_dir):
# 	os.makedirs(work_dir+save_dir)

# GCE_options = [0,1]
# GCE_tags = ["","_with_GCE_ps"]

# # for iGCE in [0,1]:
# for iGCE in [1]:
# 	model_GCE = GCE_options[iGCE]
# 	# run_tag = "king/scan_data"+GCE_tags[iGCE]
# 	run_tag = "king/scan_data_2bk4"

# 	b = 2
# 	outer = 30

# 	batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
# 				'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 				'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+\
# 				'psf_king='+str(1)+'\n'+'work_dir='+work_dir+'\n'
# 	batch3 = 'mpiexec -np 20 python np_scan.py --mask_band $mask_band --mask_ring $mask_ring '
# 	batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
# 	batch5 = '--run_tag $run_tag --psf_king $psf_king --work_dir $work_dir\n'
# 	batchn = batch1+batch2+batch3+batch4+batch5
# 	fname = './batch/'+run_tag+'_king.batch'
# 	# fname = './batch/make_bkg_'+str(emin)+'_'+str(emax)+'_model'+diff+'.batch'
# 	f=open(fname, "w")
# 	f.write(batchn)
# 	f.close()
# 	os.system("sbatch "+fname);