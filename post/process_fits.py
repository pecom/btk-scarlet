#! /home/padari/python/bin/python3.8
import numpy as np
import pandas as pd
import astropy.io.fits as fits
# import pickle5 as pickle
import os
from argparse import ArgumentParser

ddir = os.getenv("DATADIR")
odir = os.getenv("OUTDIR")

rng = np.random.default_rng()

if odir=='':
    print("Need to set output directory!")

if ddir=='':
    print("Need to set data directory!")

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-n", type=int, default=0) 
	args = parser.parse_args()
	nrun = args.n

with fits.open(f'{odir}/btk_check/blend_{nrun}.fits') as hdul:
    images = [hd.data for hd in hdul]
    image_headers = [hd.header for hd in hdul]

with fits.open(f'{odir}/btk_check/deblend_{nrun}.fits') as hdul:
    deblend_images = [hd.data for hd in hdul]
    deblend_image_headers = [hd.header for hd in hdul]

bands = list('ugrizy')
blen = len(images[0])
dlen = len(deblend_images[0])

if blen==dlen:
	print("Matching sizes. Using lazy one loop.")
# TO-DO: Separate this into two loops

for p in range(blen):
# Write all reconstructed, noisy, and noiseless images into FITS files.
	print(f"Working on image {p}...")
	blend_ims = images[0][p]
	deblend_ims = deblend_images[0][p][0] # Assuming scarlet is only giving 1 object for now
	isolated_ims = images[1][p][0] # Assuming using only 1 object from BTK for now

	for i,cb in enumerate(deblend_ims):
		hdu = fits.PrimaryHDU(cb)
		hdu.writeto(f'{odir}/fits/recon{p}_{bands[i]}.fits', overwrite=True)
	for i,cb in enumerate(blend_ims):
		hdu = fits.PrimaryHDU(cb)
		hdu.writeto(f'{odir}/fits/noisy{p}_{bands[i]}.fits', overwrite=True)
	for i,cb in enumerate(isolated_ims):
		hdu = fits.PrimaryHDU(cb)
		hdu.writeto(f'{odir}/fits/noiseless{p}_{bands[i]}.fits', overwrite=True)

# Then run source-extractor on the noisy and reconstructed files
	for b in bands:
		sys_str_recon = f"sex {odir}/fits/recon{p}_i.fits,{odir}/fits/recon{p}_{b}.fits -c noiseless_radius.se -CATALOG_NAME {odir}/cats/recon{p}_{b}.fits"
		sys_str_noisy = f"sex {odir}/fits/noisy{p}_i.fits,{odir}/fits/noisy{p}_{b}.fits -c noisy_radius.se -CATALOG_NAME {odir}/cats/noisy{p}_{b}.fits"
		sys_str_noiseless = f"sex {odir}/fits/noiseless{p}_i.fits,{odir}/fits/noiseless{p}_{b}.fits -c noiseless_radius.se -CATALOG_NAME {odir}/cats/noiseless{p}_{b}.fits"
		os.system(sys_str_recon)
		os.system(sys_str_noisy)
		os.system(sys_str_noiseless)
