import numpy as np
import pandas as pd
import astropy.io.fits as fits
# import pickle5 as pickle
import os

#with open('./bkgsub_spectra_pandaframe_solo.pkl', "rb") as fh:
#	pframe_solo = pd.read_pickle(fh)

# pframe_solo = pd.read_pickle('./bkgsub_spectra_pandaframe_solo.pkl')
pframe_solo = pd.read_pickle('../output/testSNR5_bkgsub_spectra_pandaframe_solo.pkl')
print("OPENED PANDA")

bands = list('ugrizy')
plen = len(pframe_solo)

for p in range(plen):
# Write all reconstructed and noisy images into FITS files.
	recon0 = pframe_solo['reconstruction'][p]
	noisy = pframe_solo['noisy_image'][p]
#	noisy = pframe_solo['noiseless_image'][p]
	for i,cb in enumerate(recon0):
		hdu = fits.PrimaryHDU(cb)
		hdu.writeto(f'./fits/recon{p}_{bands[i]}.fits', overwrite=True)
	for i,cb in enumerate(noisy):
		hdu = fits.PrimaryHDU(cb)
		hdu.writeto(f'./fits/noisy{p}_{bands[i]}.fits', overwrite=True)
# Then run source-extractor on the noisy and reconstructed files
	for b in bands:
		sys_str_noiseless = f"sex fits/recon{p}_i.fits,fits/recon{p}_{b}.fits -c noiseless.se -CATALOG_NAME cats/recon{p}_{b}.fits"
		sys_str_noisy = f"sex fits/noisy{p}_i.fits,fits/noisy{p}_{b}.fits -c noisy.se -CATALOG_NAME cats/noisy{p}_{b}.fits"
		os.system(sys_str_noiseless)
		os.system(sys_str_noisy)
