import os, sys
import numpy as np

batch1='''#!/bin/bash
#SBATCH -N 1   # node count
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

# file_tag_types = ["NFW_PS_below1ph_flat_2048"]
file_tag_types = ["NFW_PS_below1ph_flat_2048_dif"]
# out_tag_names = ["simple_scans/PS_below1ph_flat_dif/"]
out_tag_names = ["mpmath/"]

# b_outer_ary = [[2,15],[5,15]]
b_outer_ary = [[2,30]]

model_PSF = 0

for iGCE in [1]:
	for iROI in range(len(b_outer_ary)):
		# for MC in range(50):
		for MC in [43]:
			for ifile in range(len(file_tag_types)):
				b = b_outer_ary[iROI][0]
				outer = b_outer_ary[iROI][1]

				file_tag_type = file_tag_types[ifile]
				run_tag_base = out_tag_names[ifile]

				model_GCE = GCE_options[iGCE]
				# data_dir = "/tigress/ljchang/NPTF-IG-Check/data/simPS/MCs/with_bkg_components/flat_exposure/varyROI/b_"+str(b)+"_r_"+str(outer)+"/"
				data_dir = "/tigress/ljchang/NPTF-IG-Check/data/simPS/MCs/with_bkg_components/flat_exposure/"
				# data_file_path = data_dir+"varyROI/b_"+str(bval)+"_r_"+str(outerval)+"/"+file_tag_type+".npy"
				data_file_path = data_dir+file_tag_type+"_MC"+str(MC)+".npy"
				# run_tag = run_tag_base+"/b_"+str(b)+"_r_"+str(outer)+"/PS_dif_MC"+str(MC)
				run_tag = run_tag_base+"/PS_dif_MC"+str(MC)

				# b = 2
				# outer = 30

				batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
							'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
							'model_GCE='+str(model_GCE)+'\n'+'run_tag='+run_tag+'\n'+'model_PSF='+str(model_PSF)+'\n'+'psf_king='+str(1)+'\n'+\
							'data_file_path='+data_file_path+'\n'+'work_dir='+work_dir+'\n'
				batch3 = 'mpirun -np 20 python np_scan_simple.py --mask_band $mask_band --mask_ring $mask_ring '
				# batch3 = 'mpirun -np 20 python np_scan_1bk.py --mask_band $mask_band --mask_ring $mask_ring '
				batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --model_GCE $model_GCE '
				batch5 = '--run_tag $run_tag --model_PSF $model_PSF --psf_king $psf_king --data_file_path $data_file_path --work_dir $work_dir\n'
				batchn = batch1+batch2+batch3+batch4+batch5
				# fname = './batch/scan_'+file_tag_type+'_'+str(MC)+GCE_tags[iGCE]+'_king.batch'
				fname = './batch/'+run_tag+'.batch'
				f=open(fname, "w")
				f.write(batchn)
				f.close()
				os.system("sbatch "+fname);

