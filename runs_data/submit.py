import sys, os

batch='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=12
#SBATCH -t 25:00:00
#SBATCH --mem=30GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
#SBATCH --mail-user=sm8383@nyu.edu

source ~/.bashrc

conda activate venv_smsharma

cp -r /home/sm8383/NPTFit $SLURM_JOBTMP
cd $SLURM_JOBTMP/NPTFit
python setup.py build_ext --inplace
cd /home/sm8383/NPTF-Check/runs_data

'''

for nexp in [1]:
	for psf_king in [1]:
		for i_xsec in range(11):
			for i_mc in [0]:
				for ps_mask in ["3fgl_0p8deg"]:
					batchn = batch  + "\n"
					batchn += "mpiexec.hydra -n 12 python scan.py --nexp " + str(nexp) + " --ps_mask " + ps_mask + " --psf_king " + str(psf_king) + " --i_mc " + str(i_mc) + " --i_xsec " + str(i_xsec)
					fname = "batch/" + str(i_mc) + "_" + str(i_xsec) + "_" + str(psf_king) + "_" + ps_mask + "_" + str(nexp) + ".batch" 
					f=open(fname, "w")
					f.write(batchn)
					f.close()
					os.system("chmod +x " + fname);
					os.system("sbatch " + fname);
