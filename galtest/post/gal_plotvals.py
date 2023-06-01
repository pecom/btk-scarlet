import numpy as np
import pandas as pd
from argparse import ArgumentParser

def get_val(pf, nl, gs):
	try: 
		grecon = pd.read_pickle(f'../output/galsim_recon_nospectra_psf{pf}_sigma{nl}_galsize{gs}.pkl')
		gr_mn = np.mean(grecon['aperture_flux'])
		gr_err = quad_err(grecon['aper_err'])
	except: 
		gr_mn = 0
		gr_err = 0
	return gr_mn, gr_err

quad_err = lambda k : np.sqrt((k**2).sum())/len(k)
noise_lvls = np.array([1,2,5,10,20,30])
# noise_lvls = np.arange(1,30)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-noise", type=int, default=10)
	parser.add_argument("-psf", type=float, default=5.)
	parser.add_argument("-size", type=float, default=5.)
	args = parser.parse_args()
	given_rms = args.noise
	given_psf = args.psf
	given_gal = args.size
	if given_rms > 0:
		gr_mn, gr_err = get_val(given_psf, given_rms, given_gal)
		print(gr_mn, gr_err)
	else:
		nl_means = np.zeros((len(noise_lvls), 32))
		nl_errs = np.zeros_like(nl_means)
		for i, nl in enumerate(noise_lvls):
			gr_mn, gr_err = get_val(given_psf, nl, given_gal)
			nl_means[i,:] = gr_mn
			nl_errs[i,:] = gr_err
		np.save(f'./output/nl_means_psf{given_psf}_galsize{given_gal}.npy', nl_means)
		np.save(f'./output/nl_errs_psf{given_psf}_galsize{given_gal}.npy', nl_errs)
