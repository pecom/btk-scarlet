import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scarlet
import astropy.io.fits as fits
from argparse import ArgumentParser

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-noise", type=int, default=10)
	args = parser.parse_args()
	given_rms = args.noise
	

est_err = lambda rad_a, rad_b, rms, flx : np.sqrt(np.floor(rad_a * rad_b * np.pi) * rms**2 + flx)
frads = np.array([5,10,20])
# nmax = 130 
nmax = 60 
imtypes = list('ugrizy')

kron_fact = 1 
petro_fact = 1 

shared_keys = ["aperture_flux", "aper_err", "kron_flux", "kron_err","petro_flux", "petro_err", "kron_rad","petro_rad",  "a", "b", "x", "y"]
sex_keys = ['FLUX_APER', "FLUXERR_APER", 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_PETRO', 'FLUXERR_PETRO', "KRON_RADIUS", "PETRO_RADIUS", "A_IMAGE", "B_IMAGE", "X_IMAGE", "Y_IMAGE"]
lbl2ndx = {k:i for i,k in enumerate(sex_keys)}
noisy_data = {sk:[] for sk in shared_keys}
recon_data = {sk:[] for sk in shared_keys}



################################################################################33
# Add for loop for each file/image

counter = 0
for b in imtypes:
	for ndx in range(nmax):
		if ndx%10==0:
			print(f"On run {ndx} out of {nmax}")

		ndac = fits.open(f'./cats/noisy{b}_{ndx}.fits', memmap=False)
		# bkg_rms = float(ndac[1].data[0][0][28][10:-30])
		# if np.abs(bkg_rms - 10) > 5:
		# 	counter += 1
		# 	print("Bad rms estimate", counter)
		bkg_rms = given_rms
		noisy_vals = [ndac[2].data[sk][0] for sk in sex_keys]
		ndac.close()

		rdac = fits.open(f'./cats/recon{b}_{ndx}.fits', memmap=False)
		recon_vals = [rdac[2].data[sk][0] for sk in sex_keys]
		rdac.close()

		recon_flux = recon_vals[lbl2ndx['FLUX_APER']]
		recon_vals[1] = est_err(frads/2, frads/2, bkg_rms, recon_flux)

		im_a = recon_vals[lbl2ndx['A_IMAGE']]
		im_b = recon_vals[lbl2ndx['B_IMAGE']]

		recon_flux = recon_vals[lbl2ndx['FLUX_AUTO']]
		krad = recon_vals[lbl2ndx['KRON_RADIUS']]
		
		kerr_est = est_err(im_a*krad*kron_fact, im_b*krad*kron_fact, bkg_rms, recon_flux)
		recon_vals[3] = kerr_est
		print(f"Krad: {krad} a: {im_a} flux: {recon_flux} kerr: {kerr_est} bkg: {bkg_rms}")

		recon_flux = recon_vals[lbl2ndx['FLUX_PETRO']]
		prad = recon_vals[lbl2ndx['PETRO_RADIUS']]
		recon_vals[5] = est_err(im_a*prad*petro_fact, im_b*prad*petro_fact, bkg_rms, recon_flux)

		for i in range(len(shared_keys)):
			noisy_data[shared_keys[i]].append(noisy_vals[i])
			recon_data[shared_keys[i]].append(recon_vals[i])

noisy_frame = pd.DataFrame(data = noisy_data)
noisy_frame.to_pickle("./output/testSNR5_noisy_frame.pkl")
recon_frame = pd.DataFrame(data = recon_data)
recon_frame.to_pickle("./output/testSNR5_recon_frame.pkl")
