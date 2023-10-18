#! /home/padari/python/bin/python3.8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scarlet
import astropy.io.fits as fits
from argparse import ArgumentParser

import os

ddir = os.getenv("DATADIR")
odir = os.getenv("OUTDIR")

rng = np.random.default_rng()

if odir=='':
    print("Need to set output directory!")

if ddir=='':
    print("Need to set data directory!")
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, default=1)
    args = parser.parse_args()
    nrun = args.n
 

est_err = lambda rad_a, rad_b, rms, flx : np.sqrt(np.floor(rad_a * rad_b * np.pi) * rms**2 + flx)
# frads = np.array([5,10,20])
frads = np.arange(6,70,2)//2
# nmax = 130 
nmax = 10
imtypes = list('ugrizy')

kron_fact = 1 
petro_fact = 1 

shared_keys = ["aperture_flux", "aper_err", "kron_flux", "kron_err","petro_flux", "petro_err", "kron_rad","petro_rad",  "a", "b", "x", "y"]
sex_keys = ['FLUX_APER', "FLUXERR_APER", 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_PETRO', 'FLUXERR_PETRO', "KRON_RADIUS", "PETRO_RADIUS", "A_IMAGE", "B_IMAGE", "X_IMAGE", "Y_IMAGE"]
lbl2ndx = {k:i for i,k in enumerate(sex_keys)}
noisy_data = {sk:[] for sk in shared_keys}
noiseless_data = {sk:[] for sk in shared_keys}
recon_data = {sk:[] for sk in shared_keys}

# nless = False


################################################################################33
# Add for loop for each file/image

counter = 0
for ndx in range(nmax):
	if ndx%10==0:
		print(f"On run {ndx} out of {nmax}")
	for b in imtypes:
		noiseless_dac = fits.open(f'{odir}/cats/noiseless{ndx}_{b}.fits', memmap=False)
		noisy_dac = fits.open(f'{odir}/cats/noisy{ndx}_{b}.fits', memmap=False)
		# bkg_rms = float(ndac[1].data[0][0][28][10:-30])
		bkg_rms = float(noisy_dac[1].data[0][0][14][11:-30])
		# if np.abs(bkg_rms - 10) > 5:
			# counter += 1
			# print("Bad rms estimate", counter)
		# bkg_rms = given_rms

		noisy_vals = []
		for sk in sex_keys:
			if len(noisy_dac[2].data[sk]) > 0:
				nval = noisy_dac[2].data[sk][0]
			else:
				nval = 0
			noisy_vals.append(nval)
		# noisy_vals = [noisy_dac[2].data[sk][0] for sk in sex_keys]
		noiseless_vals = [noiseless_dac[2].data[sk][0] for sk in sex_keys]
		noisy_dac.close()
		noiseless_dac.close()

		rdac = fits.open(f'{odir}/cats/recon{ndx}_{b}.fits', memmap=False)
		recon_vals = [rdac[2].data[sk][0] for sk in sex_keys]
		rdac.close()

		recon_flux = recon_vals[lbl2ndx['FLUX_APER']]
		recon_vals[1] = est_err(frads, frads, bkg_rms, recon_flux)

		im_a = recon_vals[lbl2ndx['A_IMAGE']]
		im_b = recon_vals[lbl2ndx['B_IMAGE']]

		recon_flux = recon_vals[lbl2ndx['FLUX_AUTO']]
		krad = recon_vals[lbl2ndx['KRON_RADIUS']]
		
		kerr_est = est_err(im_a*krad*kron_fact, im_b*krad*kron_fact, bkg_rms, recon_flux)
		recon_vals[3] = kerr_est
		# print(f"Krad: {krad} a: {im_a} flux: {recon_flux} kerr: {kerr_est} bkg: {bkg_rms}")

		recon_flux = recon_vals[lbl2ndx['FLUX_PETRO']]
		prad = recon_vals[lbl2ndx['PETRO_RADIUS']]
		recon_vals[5] = est_err(im_a*prad*petro_fact, im_b*prad*petro_fact, bkg_rms, recon_flux)

		for i in range(len(shared_keys)):
			noisy_data[shared_keys[i]].append(noisy_vals[i])
			noiseless_data[shared_keys[i]].append(noiseless_vals[i])
			recon_data[shared_keys[i]].append(recon_vals[i])

noisy_frame = pd.DataFrame(data = noisy_data)
noiseless_frame = pd.DataFrame(data = noiseless_data)
recon_frame = pd.DataFrame(data = recon_data)

noisy_frame.to_pickle(f"{odir}/final/test_gal{nrun}_noisy_frame.pkl")
recon_frame.to_pickle(f"{odir}/final/test_gal{nrun}_recon_frame.pkl")
noiseless_frame.to_pickle(f"{odir}/final/test_gal{nrun}_noiseless_frame.pkl")
