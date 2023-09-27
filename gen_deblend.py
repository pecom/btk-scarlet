# -*- coding: utf-8 -*-
# Author : Prakruth Adari
# Email : prakruth.adari@stonybrook.edu

import astropy
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.table import QTable
import galsim
import matplotlib
import galcheat
import btk

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import scarlet
import sep

from scarlet.display import AsinhMapping
# from galcheat.utilities import mag2counts

# from sklearn.cluster import KMeans
from argparse import ArgumentParser
import pickle
import sys

import gen_blend

matplotlib.rc('image', interpolation='none', origin='lower')

remove = lambda x : print("REMOVE ME", *x)

class SingleDeblend():

    # TODO: UPDATE THIS COMMENT. 
    # Takes in a SINGLE image of with noise and inputted psfs.
    # Returns scarlet measured flux and model images

    def get_sepcat(self, image):
        im = image[self.indx]
        back_i = sep.Background(im)
        img_sub = im - back_i

        cata, seg = sep.extract(
                img_sub, 1.5, err=back_i.globalrms, segmentation_map=True
        )
        return cata, seg

    def __init__(self, noisy, psf, scardict={}, indx=3, scarn = 1):
       self.noisy = noisy
       # TODO: Update the assert below
       self.bands = list('ugrizy')
       self.indx = indx
       self.psfs = psf
       self.cata, self.seg = self.get_sepcat(noisy)
       self.scarlet_n = scarn

    def scarlet_getsources(self, image, centers, psfs, bkg_sub=False, indx=3):
        model_psf = scarlet.GaussianPSF(sigma=(.8,)*len(image))
        model_frame = scarlet.Frame(
            image.shape,
            psf=model_psf,
            channels=self.bands)
        rms_s = np.zeros(len(image))
        img_sub = np.zeros_like(image)

        for i, im in enumerate(image):
            bkg = sep.Background(im)
            rms_s[i] = bkg.globalrms

            # bkg_sub refers to SHOULD we do bkg subtraction, not HAVE we done bkg subtraction
            if bkg_sub:
                img_sub[i] = im - bkg
            else:
                img_sub[i] = im


        observation = scarlet.Observation(
            img_sub,
            psf=scarlet.ImagePSF(psfs),
            weights=np.ones(psfs.shape)/ (rms_s**2)[:,None,None],
            channels=self.bands).match(model_frame)

        # weights=np.ones(psfs.shape)/ (rms_s[indx]**2),
        sources, skipped = scarlet.initialization.init_all_sources(model_frame,
                                                               centers,
                                                               observation,
                                                               max_components=self.scarlet_n,
                                                               min_snr=50,
                                                               thresh=1,
                                                               fallback=True,
                                                               silent=True,
                                                               set_spectra=False
                                                              )

    #     scarlet.initialization.set_spectra_to_match(sources, observation)
        blend = scarlet.Blend(sources, observation)
        it, logL = blend.fit(1000, e_rel=1e-4)
        # Compute model
        model = blend.get_model()
        # Render it in the observed frame
        model_ = observation.render(model)
        # Compute residual
        residual = img_sub-model_
        # return sources, residual, model, observation
        return sources, residual, model, model_

    def pipeline(self, im, psfs, centered=0):
        if centered==0:
            # print("Assuming object is placed at center...")
            cent_pixval = im.shape[1]//2 - 1 
        else:
            cent_pixval = centered

        all_centers = np.array([[cent_pixval, cent_pixval]]) 
        # sc, res, mod, obs = self.scarlet_getsources(im, all_centers, psfs)
        sc, _, _, obs = self.scarlet_getsources(im, all_centers, psfs)
        scarlet_flux = scarlet.measure.flux(sc[0])
        sflux = np.array(list(scarlet_flux.data))
        return sflux, obs

    def get_deblend(self):
        noisy_flux, recon = self.pipeline(self.noisy, self.psfs)
        return noisy_flux, recon

# TODO: This is a bad skeleton of what to do with actual blended
# cases. Should think a bit more about how this should be structured
# or if we actually need 2 classes.
class MultiDeblend(SingleDeblend):
    def pipeline(self, im, psfs, centers):
        # sc, res, mod, obs = self.scarlet_getsources(im, all_centers, psfs)
        sc, _, _, obs = self.scarlet_getsources(im, centers, psfs)
        scarlet_flux = [scarlet.measure.flux(s) for s in sc]
        return scarlet_flux, obs
    
    def get_deblend(self):
        sources, obs = self.pipeline(self.noisy, self.psfs)
        return sources, obs

    # Legacy detection code
    # def get_center(self):
    #     cata = self.cata
    #     all_coords = [(cata['y'][i], cata['x'][i]) for i in range(len(cata['x']))]
    #     all_coords = np.array(all_coords)

    #     if len(all_coords) > 2:
    #         remove(["We've got more than 2 detections"])

    #     coord_norm = np.linalg.norm(all_coords - np.array([59,59]), axis=1)
    #     if np.argmin(coord_norm)==0:
    #         return all_coords
    #     else:
    #         new_coords = np.zeros_like(all_coords)
    #         new_coords[0,:] = all_coords[1,:]
    #         new_coords[1,:] = all_coords[0,:]
    #         
    #         return new_coords

def get_psfs(svy, im_shape, bd='ugrizy'):
    psfs = []
    filts = [svy.get_filter(b) for b in bd]
    pixscale = svy.pixel_scale.to_value('arcsec')
    for fp in filts:
        fwhm = fp.psf_fwhm.to_value("arcsec")
        atmos_psf = galsim.Kolmogorov(fwhm=fwhm)

        effective_wavelength = fp.effective_wavelength.to_value("angstrom")
        obscuration = svy.obscuration.value
        mirror_diameter = svy.mirror_diameter.to_value("m")
        lam_over_diam = 3600 * np.degrees(1e-10 * effective_wavelength / mirror_diameter)
        optical_psf_model = galsim.Airy(lam_over_diam=lam_over_diam, obscuration=obscuration)

        fin_psf = galsim.Convolve(atmos_psf, optical_psf_model)

        psfs.append(fin_psf.drawImage(nx=im_shape[0], ny=im_shape[1], scale=pixscale).array)
        # psfs.append(fin_psf.drawImage(galsim.Image(*im_shape), scale=pixscale).array)
    psfs = np.array(psfs)

    return psfs

def lazy_psfs(svy, im_shape, bd='ugrizy'):
    psfs = []
    for sf in range(len(bd)):
        psfs.append(svy.filters[sf].psf.drawImage(bounds=galsim.BoundsI(xmin=1, xmax=im_shape[0], ymin=1, ymax=im_shape[1])).array)
    psfs = np.array(psfs)

    return psfs


def compare_flux(solved, true):
    N = len(solved)
    f = len(solved[0])
    ratios = np.zeros(f)
    err = np.zeros(f)

    for i in range(f):
        ratios[i] = np.mean([solv[i] for solv in solved])/true[0][i]
        err[i] = np.std([solv[i] for solv in solved])/(true[0][i] * np.sqrt(N))
    
    return ratios, err

def scarlet_single(df, input_df, ndx=4, verbose=False, scar_N=1, bsize=10):
    
    surveys = galcheat.get_surveys("LSST")
    psfs = get_psfs(surveys, (120,120))
    
    # surveys = btk.survey.get_surveys("Rubin")
    # psfs = lazy_psfs(surveys, (120,120))
    bands = list('ugrizy')
    

    for bs in range(bsize):
        noisy_row = input_df.loc[(input_df['cata_ndx'] == ndx) * (input_df['run_ndx'] == bs)]
        noisy_im = noisy_row['noisy_image'].values[0]
        if verbose:
            print(f"Deblending run {bs}...")
        sblend = SingleDeblend(noisy_im, psfs, scarn=scar_N)
        flux, recon = sblend.get_deblend()
        
        df["cata_ndx"].append(ndx)
        df["run_ndx"].append(bs)
        df["scarlet_flux"].append(flux)
        df["reconstruction"].append(recon)
    
    # Maybe think about if I want to modify df inline or something else
    return None

# def scarlet_double(df, shift=4, saveFiles=True, verbose=False, bsize=10):
#     # This hasn't been cleaned or tested.
    
#     suvy = galcheat.get_survey("LSST")
#     psfs = get_psfs(surveys, (120, 120))

#     true_mags = np.array([catalog.table[shift][ab] for ab in ['u_ab', 'g_ab', 'r_ab', 'i_ab', 'z_ab', 'y_ab']])

#     for bs in range(bsize):
#         noisy_im = blend_images[bs]
#         noiseless_im = blend_iso[bs][0]
#         sblend = MultiDeblend(noisy_im, psfs)
#         true_flux = sblend.lsst_mags_adu(true_mags, bands, suvy)
#         res = sblend.get_deblend()
#         # print(res, true_flux)
#         df["noisy_flux"].append(res[0])
#         df["reconstruction"].append(res[2])
#     if verbose:
#         print("Got fluxes")

#     return None

def get_schema():
    dframe_keys = ["cata_ndx", "run_ndx", "scarlet_flux", "reconstruction"]
    dframe_solo = {sk:[] for sk in dframe_keys}
    return dframe_solo