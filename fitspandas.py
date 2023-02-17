import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scarlet
import astropy.io.fits as fits

est_err = lambda rad_a, rad_b, rms, flx : np.sqrt(np.floor(rad_a * rad_b * np.pi) * rms**2 + flx)
frads = np.array([5,10,20])
# nmax = 130 
nmax = 60 
bands = list('ugrizy')

shared_keys = ["aperture_flux", "aper_err", "petro_flux", "petro_err", "kron_flux", "kron_err", "petro_rad", "kron_rad", "a", "b"]
sex_keys = ['FLUX_APER', "FLUXERR_APER", 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_PETRO', 'FLUXERR_PETRO', "KRON_RADIUS", "PETRO_RADIUS", "A_IMAGE", "B_IMAGE"]
noisy_data = {sk:[] for sk in shared_keys}
recon_data = {sk:[] for sk in shared_keys}



################################################################################33
# Add for loop for each file/image

for ndx in range(nmax):
	if ndx%10==0:
		print(f"On run {ndx} out of {nmax}")

	for b in bands:

		ndac = fits.open(f'./cats/noisy{ndx}_{b}.fits', memmap=False)
		bkg_rms = float(ndac[1].data[0][0][14][11:-30])
		noisy_vals = [ndac[2].data[sk][0] for sk in sex_keys]
		ndac.close()

		rdac = fits.open(f'./cats/recon{ndx}_{b}.fits', memmap=False)
		recon_vals = [rdac[2].data[sk][0] for sk in sex_keys]
		rdac.close()

		recon_flux = recon_vals[0]
		recon_vals[1] = est_err(frads/2, frads/2, bkg_rms, recon_flux)

		recon_flux = recon_vals[2]
		krad = recon_vals[-4]
		recon_vals[3] = est_err(recon_vals[-1]*krad, recon_vals[-2]*krad, bkg_rms, recon_flux)

		recon_flux = recon_vals[4]
		prad = recon_vals[-3]
		recon_vals[5] = est_err(recon_vals[-1]*prad, recon_vals[-2]*prad, bkg_rms, recon_flux)

		for i in range(len(shared_keys)):
			noisy_data[shared_keys[i]].append(noisy_vals[i])
			recon_data[shared_keys[i]].append(recon_vals[i])


noisy_frame = pd.DataFrame(data = noisy_data)
noisy_frame.to_pickle("./output/testSNR5_noisy_frame.pkl")
recon_frame = pd.DataFrame(data = recon_data)
recon_frame.to_pickle("./output/testSNR5_recon_frame.pkl")
