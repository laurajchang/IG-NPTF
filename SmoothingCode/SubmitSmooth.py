import sys, os
import numpy as np
# from local_dirs import *

batch1='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
#SBATCH --mem=4GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=ljchang@princeton.edu

cd /tigress/ljchang/NPTF-IG-Check/SmoothingCode/
'''

data_dir = "/tigress/ljchang/NPTF-IG-Check/data/simPS/"
save_dir = "/tigress/ljchang/NPTF-IG-Check/data/simPS/smoothedMaps/"

data_tag = "NFW_PS_below1ph_flat_512"
data_file_path = data_dir+data_tag+".npz"

ebin = 10
save_tag = data_tag+"_bin"+str(ebin)

batch2 = 'data_file_path='+data_file_path+'\n'+'save_dir='+save_dir+'\n'+"save_tag="+save_tag+"\n"+"ebin="+str(ebin)+"\n"
batch3 = 'python SmoothInterface.py --data_file_path $data_file_path --save_dir $save_dir '
batch4 = '--save_tag $save_tag --ebin $ebin\n'
batchn = batch1+batch2+batch3+batch4
# fname = './batch/scan_'+file_tag_type+'_'+str(MC)+GCE_tags[iGCE]+'_king.batch'
fname = './batch/smooth_'+save_tag+'.batch'
f=open(fname, "w")
f.write(batchn)
f.close()
os.system("sbatch "+fname);
