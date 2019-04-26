import os, sys
import numpy as np

batch1='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:05:00
#SBATCH --mem=24GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=ljchang@princeton.edu

cd /tigress/ljchang/NPTF_test/Bkg-Maps
		'''

#############################################
# For getting best-fit Poissonian templates #
#############################################

work_dir = '/tigress/ljchang/NPTF_test/Bkg-Maps/'

# for model in ['B','C']:
# for model in ['p6','p7','p8']:
for model in ['p6']:
	save_dir = '/templates_example/'

	# Make save dir if it doesn't exist
	if not os.path.exists(work_dir+save_dir):
		os.makedirs(work_dir+save_dir)

	band_vals = [2]
	arb_ind = 0
	for b in band_vals:
		outer_vals = [30]
		# outer_vals = [40]
		for outer in outer_vals:
			batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+ \
						'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
						'save_dir='+work_dir+save_dir+'\n'
			batch3 = 'python make_fermi_background_example.py --mask_band $mask_band --mask_ring $mask_ring '
			batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval '
			batch5 = '--save_dir $save_dir\n'
			batchn = batch1+batch2+batch3+batch4+batch5
			fname = './batch/make_bkg_b_'+str(b)+'_r_'+str(outer)+'_example.batch'
			# fname = './batch/make_bkg_'+str(emin)+'_'+str(emax)+'_model'+diff+'.batch'
			f=open(fname, "w")
			f.write(batchn)
			f.close()
			os.system("sbatch "+fname);
			arb_ind += 1

# work_dir = '/tigress/smsharma/Fermi-SmoothGalHalo/DataFiles/Bkg-Maps/'

# b = 20
# outer = 45
# for model in ['p7']:
# 	for b_ind in range(len(hemi_b)):
# 		save_dir = '/templates_'+model+'_decay/b_'+str(b)+'_r_'+str(outer)+'_'+hemi_b[b_ind]

# 		# Make save dir if it doesn't exist
# 		if not os.path.exists(work_dir+save_dir):
# 			os.makedirs(work_dir+save_dir)

# 		diff = model

# 		batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+'mask_b='+str(1)+'\n'+ \
# 					'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 					'mask_bmin='+str(bmin_vals[b_ind])+'\n'+'mask_bmax='+str(bmax_vals[b_ind])+'\n'+ \
# 					'hemi_b='+'_'+hemi_b[b_ind]+'\n'+'emin='+str(emin)+'\n'+'emax='+str(emax)+'\n'+'diff='+diff+'\n'+'save_dir='+work_dir+save_dir+'\n'
# 		batch3 = 'python make_fermi_background.py --mask_band $mask_band --mask_ring $mask_ring --mask_b $mask_b '
# 		batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --mask_bmin $mask_bmin --mask_bmax $mask_bmax '
# 		batch5 = '--hemi_b $hemi_b --emin $emin --emax $emax --diff $diff --save_dir $save_dir\n'
# 		batchn = batch1+batch2+batch3+batch4+batch5
# 		fname = './batch/make_bkg_b_'+str(b)+'_r_'+str(outer)+'_'+str(emin)+'_'+str(emax)+'_'+diff+'_'+hemi_b[b_ind]+'.batch'
# 		f=open(fname, "w")
# 		f.write(batchn)
# 		f.close()
# 		os.system("sbatch "+fname);

# work_dir = '/tigress/smsharma/Fermi-SmoothGalHalo/DataFiles/Bkg-Maps/'

# for model in ['A','B','C']:
# 	save_dir = '/templates_model'+model+'_S/'

# 	# Make save dir if it doesn't exist
# 	if not os.path.exists(work_dir+save_dir):
# 		os.makedirs(work_dir+save_dir)

# 	diff = model
# 	band_vals = [20]
# 	arb_ind = 0
# 	for b in band_vals:
# 		# outer_vals = [50,100,150]
# 		outer_vals = [50]
# 		for outer in outer_vals:
# 			for b_ind in [1]:
# 				batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+'mask_b='+str(1)+'\n'+ \
# 							'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 							'mask_bmin='+str(bmin_vals[b_ind])+'\n'+'mask_bmax='+str(bmax_vals[b_ind])+'\n'+ \
# 							'hemi_b='+'_'+hemi_b[b_ind]+'\n'+'emin='+str(emin)+'\n'+'emax='+str(emax)+'\n'+'diff='+diff+'\n'+'save_dir='+work_dir+save_dir+'\n'
# 				batch3 = 'python make_fermi_background.py --mask_band $mask_band --mask_ring $mask_ring --mask_b $mask_b '
# 				batch4 = '--mask_bandval $mask_bandval --mask_outerval $mask_outerval --mask_bmin $mask_bmin --mask_bmax $mask_bmax '
# 				batch5 = '--hemi_b $hemi_b --emin $emin --emax $emax --diff $diff --save_dir $save_dir\n'
# 				batchn = batch1+batch2+batch3+batch4+batch5
# 				fname = './batch/make_bkg_'+str(emin)+'_'+str(emax)+'_model'+diff+'.batch'
# 				f=open(fname, "w")
# 				f.write(batchn)
# 				f.close()
# 				os.system("sbatch "+fname);
# 				arb_ind += 1


# band_vals = [20]
# inner = 0
# arb_ind = 0
# diff = 'p6'

# for b in band_vals:
# 	# outer_vals = np.arange(b+25,180-b+5,5)
# 	# outer_vals = [50,100,150]
# 	outer_vals = [50]
# 	for outer in outer_vals:
# 	    for b_ind in range(len(bmin_vals)):
#     	# for b_ind in [1]:
# 			batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+'mask_b='+str(1)+'\n'+ \
# 						'mask_bandval='+str(b)+'\n'+'mask_innerval='+str(inner)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 						'mask_bmin='+str(bmin_vals[b_ind])+'\n'+'mask_bmax='+str(bmax_vals[b_ind])+'\n'+ \
# 						'hemi_b='+'_'+hemi_b[b_ind]+'\n'+'emin='+str(emin)+'\n'+'emax='+str(emax)+'\n'+'diff='+diff+'\n'+'save_dir='+work_dir+save_dir+'\n'
# 			batch3 = 'python make_fermi_background.py --mask_band $mask_band --mask_ring $mask_ring --mask_b $mask_b '
# 			batch4 = '--mask_bandval $mask_bandval --mask_innerval $mask_innerval --mask_outerval $mask_outerval --mask_bmin $mask_bmin --mask_bmax $mask_bmax '
# 			batch5 = '--hemi_b $hemi_b --emin $emin --emax $emax --diff $diff --save_dir $save_dir\n'
# 			batchn = batch1+batch2+batch3+batch4+batch5
# 			fname = './batch/make_bkg_'+str(emin)+'_'+str(emax)+'_'+str(arb_ind)+hemi_b[b_ind]+'.batch'
# 			f=open(fname, "w")
# 			f.write(batchn)
# 			f.close()
# 			os.system("sbatch "+fname);
# 			arb_ind += 1

# band_vals = [15]
# inner = 20
# arb_ind = 0
# diff = 'p6'

# for b in band_vals:
# 	# outer_vals = np.arange(b+25,180-b+5,5)
# 	# outer_vals = [50,100,150]
# 	outer_vals = [50]
# 	for outer in outer_vals:
# 	    for b_ind in range(len(bmin_vals)):
# 	    	for l_ind in range(len(lmin_vals)):
# 				batch2 = 'mask_band='+str(1)+'\n'+'mask_ring='+str(1)+'\n'+'mask_b='+str(1)+'\n'+'mask_l='+str(1)+'\n'+ \
# 							'mask_bandval='+str(b)+'\n'+'mask_innerval='+str(inner)+'\n'+'mask_outerval='+str(outer)+'\n'+ \
# 							'mask_bmin='+str(bmin_vals[b_ind])+'\n'+'mask_bmax='+str(bmax_vals[b_ind])+'\n'+ \
# 							'mask_lmin='+str(lmin_vals[l_ind])+'\n'+'mask_lmax='+str(lmax_vals[l_ind])+'\n'+ \
# 							'hemi_b='+'_'+hemi_b[b_ind]+'\n'+'hemi_l='+hemi_l[l_ind]+'\n'+'emin='+str(emin)+'\n'+'emax='+str(emax)+'\n'+'diff='+diff+'\n'+'save_dir='+work_dir+save_dir+'\n'
# 				batch3 = 'python make_fermi_background.py --mask_band $mask_band --mask_ring $mask_ring --mask_b $mask_b --mask_l $mask_l '
# 				batch4 = '--mask_bandval $mask_bandval --mask_innerval $mask_innerval --mask_outerval $mask_outerval --mask_bmin $mask_bmin --mask_bmax $mask_bmax --mask_lmin $mask_lmin --mask_lmax $mask_lmax '
# 				batch5 = '--hemi_b $hemi_b --hemi_l $hemi_l --emin $emin --emax $emax --diff $diff --save_dir $save_dir\n'
# 				batchn = batch1+batch2+batch3+batch4+batch5
# 				fname = './batch/make_bkg_'+str(emin)+'_'+str(emax)+'_'+str(arb_ind)+hemi_b[b_ind]+hemi_l[l_ind]+'.batch'
# 				f=open(fname, "w")
# 				f.write(batchn)
# 				f.close()
# 				os.system("sbatch "+fname);
# 				arb_ind += 1

# arb_ind = 0
# b = 20
# inner = 0
# outer = 50
# diff='p6'
# for mask_eighths_ind in range(8):
# 	batch2 = 'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+'mask_innerval='+str(inner)+'\n'+'mask_eighths='+str(1)+'\n'+'mask_eighths_ind='+str(mask_eighths_ind)+'\n'+ \
# 			 'emin='+str(emin)+'\n'+'emax='+str(emax)+'\n'+'diff='+diff+'\n'+'save_dir='+work_dir+save_dir+'\n'
# 	batch3 = 'python make_fermi_background.py --mask_bandval $mask_bandval --mask_innerval $mask_innerval --mask_outerval $mask_outerval '
# 	batch4 = '--mask_eighths $mask_eighths --mask_eighths_ind $mask_eighths_ind '
# 	batch5 = '--emin $emin --emax $emax --diff $diff --save_dir $save_dir\n'
# 	batchn = batch1+batch2+batch3+batch4+batch5
# 	fname = './batch/make_bkg_'+str(emin)+'_'+str(emax)+'_eighths_'+str(mask_eighths_ind)+'.batch'
# 	f=open(fname, "w")
# 	f.write(batchn)
# 	f.close()
# 	os.system("sbatch "+fname);
# 	arb_ind += 1

# arb_ind = 0
# b = 20
# inner = 0
# outer = 50
# diff='p6'
# for mask_sixteenths_ind in range(16):
# 	batch2 = 'mask_bandval='+str(b)+'\n'+'mask_outerval='+str(outer)+'\n'+'mask_innerval='+str(inner)+'\n'+'mask_sixteenths='+str(1)+'\n'+'mask_sixteenths_ind='+str(mask_sixteenths_ind)+'\n'+ \
# 			 'emin='+str(emin)+'\n'+'emax='+str(emax)+'\n'+'diff='+diff+'\n'+'save_dir='+work_dir+save_dir+'\n'
# 	batch3 = 'python make_fermi_background.py --mask_bandval $mask_bandval --mask_innerval $mask_innerval --mask_outerval $mask_outerval '
# 	batch4 = '--mask_sixteenths $mask_sixteenths --mask_sixteenths_ind $mask_sixteenths_ind '
# 	batch5 = '--emin $emin --emax $emax --diff $diff --save_dir $save_dir\n'
# 	batchn = batch1+batch2+batch3+batch4+batch5
# 	fname = './batch/make_bkg_'+str(emin)+'_'+str(emax)+'_sixteenths_'+str(mask_sixteenths_ind)+'.batch'
# 	f=open(fname, "w")
# 	f.write(batchn)
# 	f.close()
# 	os.system("sbatch "+fname);
# 	arb_ind += 1
