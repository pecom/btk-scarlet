import numpy as np
import pandas as pd
from argparse import ArgumentParser

def get_val(pf, nl, gs):
	try: 
		grecon = pd.read_pickle(f'../output/galsim_noisy_nospectra_psf{pf}_sigma{nl}_galsize{gs}.pkl')
		grecon = pd.read_pickle(f'../output/galsim_noisy_nospectra_psf{pf}_sigma{nl}_galsize{gs}.pkl')
		gr_mn = grecon['kron_rad']
		gr_a = grecon['a']
		gr_b = grecon['b']
	except: 
		gr_mn = 0
		gr_a = 0
		gr_b = 0
	return gr_mn, gr_a, gr_b

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
		gr_mn, gr_a, gr_b = get_val(given_psf, given_rms, given_gal)
		gr_kron = np.mean(gr_mn)
		print(f"Kron radius : {gr_kron} {np.std(gr_mn)} or scaled {gr_kron *np.sqrt(np.mean(gr_a) * np.mean(gr_b))}")
	else:
		nl_means = np.zeros(len(noise_lvls))
		for i, nl in enumerate(noise_lvls):
			gr_mn, gr_a, gr_b = get_val(given_psf, nl, given_gal)
			gr_kron = np.mean(gr_mn) * np.sqrt(np.mean(gr_a) * np.mean(gr_b))
			nl_means[i] = gr_kron
		print(nl_means)
		if (nl_means==0).all():
			print("All 0s skipping")
		else:
			np.save(f'./output/nl_kron_psf{given_psf}_galsize{given_gal}.npy', nl_means)
