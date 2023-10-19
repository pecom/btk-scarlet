#! /home/padari/python/bin/python3.8

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
import astropy.io.fits as fits

import os
import sys
import btk
from argparse import ArgumentParser


odir = os.getenv("OUTDIR")

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-n", type=int)
	args = parser.parse_args()
	ndx = args.n

	recon_frame = pd.read_pickle(f'{odir}/final//test_gal{ndx}_recon_frame.pkl')
	noisy_frame = pd.read_pickle(f'{odir}/final/test_gal{ndx}_noisy_frame.pkl')
	noiseless_frame = pd.read_pickle(f'{odir}/final/test_gal{ndx}_noiseless_frame.pkl')
	deblend_batch = btk.blend_batch.DeblendBatch.load_fits(f'{odir}/btk_check/', ndx)
	frads = np.arange(6,70,2)//2

	true_aper = [noiseless_frame['aperture_flux'][i][10] for i in range(6)]

	recon_aper = np.array(recon_frame['aperture_flux'].values.tolist())
	noisy_aper = np.array(noisy_frame['aperture_flux'].values.tolist())
	
	print(f"Recon aperture: {recon_aper.shape}")

	test_deblend = deblend_batch.deblended_images.byteswap().newbyteorder()
	center_pixel = test_deblend.shape[-1]//2 - 1
	centers = np.repeat([center_pixel], 10)
	tcent = centers.reshape((10,1,))

	sep_ratios = np.zeros((10, len(frads), 6))
	color_diffs = np.zeros((10, len(frads), 5))

	kron_rad = np.mean(recon_frame['kron_rad']/2 * np.sqrt(recon_frame['a'] * recon_frame['b']))

	for run_ndx in range(10):

		test_im = test_deblend[run_ndx][0]
		recon_ndx = recon_aper[(run_ndx)*6:(run_ndx+1)*6]

		for i,fr in enumerate(frads):
			btk_aper = btk.measure.get_aperture_fluxes(test_im, xs=tcent, ys=tcent, radius=fr, sky_level=1)[:,0]
			sep_ratios[run_ndx,i,:] = btk_aper/recon_ndx[:,i]

		for i,fr in enumerate(frads):
			btk_aper = btk.measure.get_aperture_fluxes(test_im, xs=tcent, ys=tcent, radius=fr, sky_level=1)[:,0]
			btk_mag = np.log(btk_aper[:-1]/btk_aper[1:])
			sex_mag = np.log(recon_ndx[:,i][:-1]/recon_ndx[:,i][1:])
			color_diffs[run_ndx,i,:] = btk_mag - sex_mag

	bands = 'ugrizy'
	colors = [bands[:-1][i]+bands[1:][i] for i in range(5)]

	fig, ax = plt.subplots(1,2, figsize=(10,5))

	ax[0].plot(frads, np.mean(sep_ratios, axis=(0,2)), '.')
	ax[0].axhline(y=1, ls='--', color='black')
	ax[0].set_title("Fluxes")
	ax[0].axvline(x=kron_rad, ls='--', color='black')

	ax[1].plot(frads, np.mean(color_diffs, axis=(0,2)), '.')
	ax[1].axhline(y=0, ls='--', color='black')
	ax[1].set_title("Colors Difference")
	ax[1].axvline(x=kron_rad, ls='--', color='black')

	fig.suptitle(f"Mean of SExtractor vs SEP Aperture of Galaxy {ndx}")
	plt.tight_layout()
	plt.savefig(f"{odir}/plots/sep_sex_galaxy{ndx}.png")

	fig, ax = plt.subplots(1,2, figsize=(12,6))

	for i,b in enumerate(bands):
	    ax[0].plot(frads, np.mean(sep_ratios, axis=0)[:,i], '.', label=b)
	ax[0].axhline(y=1, ls='--', color='black')
	ax[0].set_title("Fluxes")
	ax[0].legend(frameon=False)
	ax[0].axvline(x=kron_rad, ls='--', color='black')

	for i,c in enumerate(colors):
	    ax[1].plot(frads, np.mean(color_diffs, axis=0)[:,i], '.-', label=c)
	ax[1].axhline(y=0, ls='--', color='black')
	ax[1].set_title("Colors Difference")
	ax[1].legend(frameon=False)
	ax[1].axvline(x=kron_rad, ls='--', color='black')

	fig.suptitle("Per band SExtractor vs SEP Aperture")
	plt.tight_layout()
	plt.savefig(f"{odir}/plots/sep_sex_band_galaxy{ndx}.png")
	np.save(f"{odir}/plots/data/data_sep_sex_galaxy{ndx}_flux", sep_ratios )
	np.save(f"{odir}/plots/data/data_sep_sex_galaxy{ndx}_color", color_diffs )




