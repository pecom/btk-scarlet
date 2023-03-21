import os
import galsim 
import numpy as np
import scarlet
import sep



def make_gal():
	# Basic run of the mill galsim properties
	gal_flux = 1.e5
	gal_sigma = 2.
	psf_sigma = 1.
	pixel_scale = 0.2
	noise = 10.

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


def scarlet_deblend(image, bkg_sub=True, rms_est = 10):
    centers = np.array([[66,66]])
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
	    psf=scarlet.GaussianPSF(sigma=5),
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
							       set_spectra=True
							      )

	#     scarlet.initialization.set_spectra_to_match(sources, observation)
	blend = scarlet.Blend(sources, observation)
	it, logL = blend.fit(1000, e_rel=1e-4)
    print(f"{logL} after {it} iterations")
	# Compute model
	model = blend.get_model()
	# Render it in the observed frame
	model_ = observation.render(model)
	# Compute residual
	residual = img_sub-model_
	# return sources, residual, model, observation
	return sources, residual, model, model_


for i in range(100):
	if i%10==0:
		print(f"On run {i}/100")
	ima, ims, psfs = make_gal()
	ima.write(f'./fits/noisy1_{i}.fits')
	_, _, _, md = scarlet_deblend(np.array([ims]))
	scar_fin = galsim.Image(md[0], copy=True)
	scar_fin.write(f'./fits/demo1_{i}.fits')

print("All done! Made MD")
