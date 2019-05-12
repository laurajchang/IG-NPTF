import numpy as np
import healpy as hp

class simPS:

	def __init__(self, f_ary, dNdF_ary, spatial_map, nsources, nside=128, verbose=False):
		""" Class that takes in a sampled source-count function and a spatial map, which specify the flux PDF 
		number density spatial PDF, respectively, and generates point sources (PSs).

		:param f_ary: Array of flux values at which source-count is sampled.
		:param dNdF_ary: Array of source-count values.
		:param spatial_map: Healpix map specifying spatial distribution of PSs, with nside=nside.
		:param nsources: Number of PSs to generate. 
		"""
		self.f_ary = f_ary
		self.dNdF_ary = dNdF_ary
		self.spatial_map = spatial_map
		self.nsources = nsources
		self.nside = nside
		self.verbose = verbose

		self.npix = hp.nside2npix(self.nside)

		self.make_map()
		if self.verbose:
			print("Generating a map of",self.nsources,"point sources with nside =",self.nside)
	def sample_fluxes(self):
		df_ary = np.zeros(len(self.f_ary))
		df_ary[1:] = self.f_ary[1:]-self.f_ary[:-1]
		inds, cdf = CDF(df_ary,self.dNdF_ary)
		self.fluxvals = PDFSample(self.f_ary,inds,cdf,self.nsources)

	def sample_pixels(self):
		pixvals = np.arange(self.npix)
		inds, cdf = CDF(np.ones(self.npix),self.spatial_map)
		self.pixvals = PDFSample(pixvals,inds,cdf,self.nsources)

	def make_map(self):
		self.sample_fluxes()
		self.sample_pixels()

		self.flux_map = np.zeros(self.npix)
		for i in range(self.nsources):
			self.flux_map[self.pixvals[i]] += self.fluxvals[i]

def CDF(dxvals,pofx):
	yvals = pofx*dxvals
	sortxvals = np.argsort(yvals)
	yvals = yvals[sortxvals]
	return sortxvals,np.cumsum(yvals)

def PDFSample(xvals,sortxvals,cdf,samples):
	unidraw = np.random.uniform(high=cdf[-1], size=samples)
	cdfdraw = np.searchsorted(cdf, unidraw)
	cdfdraw = sortxvals[cdfdraw]
	return xvals[cdfdraw]