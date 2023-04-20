#! /home/padari/python/bin/python3.8

import os
import galsim 
import numpy as np
import scarlet
import sep
from argparse import ArgumentParser


def make_gal(noise, p_sigma, g_sigma):
	# Basic run of the mill galsim properties
	gal_flux = 1.e5
	gal_sigma = g_sigma
	psf_sigma = p_sigma
	pixel_scale = 0.2

	psf_scale = psf_sigma/pixel_scale

	gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)
	psf = galsim.Gaussian(flux=1., sigma=psf_sigma)

	final = galsim.Convolve([gal, psf])

	image = final.drawImage(scale=pixel_scale)
	image.addNoise(galsim.GaussianNoise(sigma=noise))

	return image, image.array, psf

def sep_detect(im):
	bkg = sep.Background(im)
	cata = sep.extract(im - bkg, 2.5, err=bkg.globalrms)
	
	return cata


def scarlet_deblend(image, obs_sig, bkg_sub=True, rms_est = 10, verbose=False):
	# centers = np.array([[66,66]])
	c_pix = image.shape[1]//2
	centers = np.array([[c_pix, c_pix]])
	model_psf = scarlet.GaussianPSF(sigma=.8)
	model_frame = scarlet.Frame(
			image.shape,
			psf=model_psf,
			channels=list('i'))
	rms_s = np.zeros(len(image))
	img_sub = np.zeros_like(image)

	if bkg_sub:
		for i, im in enumerate(image):
			bkg = sep.Background(im)
			img_sub[i] = im - bkg
			rms_s[i] = bkg.globalrms
	else:
		for i,im in enumerate(image):
			img_sub[i] = im
			rms_s[i] = rms_est

	observation = scarlet.Observation(
		img_sub,
		psf=scarlet.GaussianPSF(sigma=obs_sig),
		weights=np.ones(image.shape)/ (rms_s**2)[:,None,None],
		channels=list('i')).match(model_frame)

	# weights=np.ones(psfs.shape)/ (rms_s[indx]**2),
	sources, skipped = scarlet.initialization.init_all_sources(model_frame,
							       centers,
							       observation,
							       max_components=1,
							       min_snr=50,
							       thresh=1,
							       fallback=False,
							       silent=True,
							       set_spectra=False
							      )

	#     scarlet.initialization.set_spectra_to_match(sources, observation)
	blend = scarlet.Blend(sources, observation)
	it, logL = blend.fit(1000, e_rel=1e-4)
	if verbose: print(f"{logL} after {it} iterations")
	# Compute model
	model = blend.get_model()
	# Render it in the observed frame
	model_ = observation.render(model)
	# Compute residual
	residual = img_sub-model_
	# return sources, residual, model, observation
	return sources, residual, model, model_


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-noise", type=int, default=10)
	parser.add_argument("-psf", type=float, default=5.)
	parser.add_argument("-size", type=float, default=5.)
	args = parser.parse_args()
	N = 100
	for i in range(N):
		if i%10==0:
			print(f"On run {i}/{N}")
		run_psf = args.psf/10
		pix_psf = run_psf/.2
		if i==0:
			print(f"Running with params {args.noise} {run_psf} {args.size}")
		ima, ims, psfs = make_gal(args.noise, run_psf, args.size)
		ima.write(f'./fits/noisy1_{i}.fits')
		_, res, _, md = scarlet_deblend(np.array([ims]), pix_psf, bkg_sub=False, rms_est=args.noise)
		scar_fin = galsim.Image(md[0], copy=True)
		scar_fin.write(f'./fits/demo1_{i}.fits')
		scar_res = galsim.Image(res[0], copy=True)
		scar_res.write(f'./fits/resid1_{i}.fits')

	print("All done! Made MD")
