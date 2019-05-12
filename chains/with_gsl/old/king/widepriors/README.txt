Wider priors for 2-break scan middle slope, e.g. 

n.add_non_poiss_model('iso_ps',
		  ['$\log_{10}(A_\mathrm{iso}^\mathrm{ps})$','$n_1^\mathrm{iso}$','$n_2^\mathrm{iso}$','$n_3^\mathrm{iso}$','$S_b^{(1),\mathrm{iso}}$','$S_b^{(2),\mathrm{iso}}$'],
		  [[-10,2],[2.05,15.],[-2.95,1.95],[-1.95,0.95],[0.7,3.5],[-2,0.5]],
		  [True,False,False,False,True,True])

