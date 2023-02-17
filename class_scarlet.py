#!/usr/bin/env python
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

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import scarlet
import sep

import btk
import btk.plot_utils
import btk.survey
import btk.draw_blends
import btk.catalog
import btk.sampling_functions
from matplotlib.patches import Ellipse
import astropy.visualization

from scarlet.display import AsinhMapping
from galcheat.utilities import mag2counts

from sklearn.cluster import KMeans
import pickle
import sys

matplotlib.rc('image', interpolation='none', origin='lower')

print("Done importing")


remove = lambda x : print("REMOVE ME", *x)

class CenteredSampling(btk.sampling_functions.SamplingFunction):
    
    def __init__(self, stamp_size=24.0, maxshift=None, setndx = 24):
        super().__init__(1)
        self.stamp_size = stamp_size
        self.maxshift = maxshift if maxshift else self.stamp_size / 10.0
        self.setndx = setndx

    def _get_random_center_shift(self, num_objects, maxshift):
        x_peak = np.random.uniform(-maxshift, maxshift, size=num_objects)
        y_peak = np.random.uniform(-maxshift, maxshift, size=num_objects)
        return x_peak, y_peak
        
    @property
    def compatible_catalogs(self):
        return "CatsimCatalog", "CosmosCatalog"

    def __call__(self,table):
        
        indexes = [self.setndx]
        blend_table = table[indexes]
        
        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0
        

        if np.any(blend_table["ra"] > self.stamp_size / 2.0) or np.any(
            blend_table["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_table

class PairSampling(btk.sampling_functions.SamplingFunction):
    
    def __init__(self, stamp_size=24.0, maxshift=None, setshift=None, ndx=24):
        super().__init__(2)
        self.stamp_size = stamp_size
        self.maxshift = maxshift if maxshift else self.stamp_size / 10.0
        self.setshift = setshift if setshift else 5
        self.ndx = ndx

    def _get_random_center_shift(self, num_objects, maxshift):
        x_peak = np.random.uniform(-maxshift, maxshift, size=num_objects)
        y_peak = np.random.uniform(-maxshift, maxshift, size=num_objects)
        return x_peak, y_peak
        
    @property
    def compatible_catalogs(self):
        return "CatsimCatalog", "CosmosCatalog"

    def __call__(self,table):
        
        indexes = [self.ndx,self.ndx]
        blend_table = table[indexes]
        
        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0
        
        # x_peak, y_peak = self._get_random_center_shift(1, self.maxshift)
        rand_theta = np.random.uniform(0, 2*np.pi)
        x_peak = self.setshift * np.cos(rand_theta)
        y_peak = self.setshift * np.sin(rand_theta)
        
        blend_table["ra"][1] += x_peak
        blend_table["dec"][1] += y_peak

        if np.any(blend_table["ra"] > self.stamp_size / 2.0) or np.any(
            blend_table["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_table

class SingleDeblend():

    # Takes in a SINGLE image of (6, 120, 120) with noise and
    # provide psfs for scarlet. Can be done in the class but outsourced
    # to prevent redoing the same creation multiple times.

    def get_sepcat(self, image):
        im = image[self.indx]
        back_i = sep.Background(im)
        img_sub = im - back_i

        cata, seg = sep.extract(
                img_sub, 1.5, err=back_i.globalrms, segmentation_map=True
        )
        return cata, seg

    def __init__(self, noisy, psf, scardict={}, indx=3):
       self.noisy = noisy
       self.bands = list('ugrizy')
       self.indx = indx
       self.psfs = psf
       self.cata, self.seg = self.get_sepcat(noisy)

    def comp_flux(self, sources):
        scarlet_flux = scarlet.measure.flux(sources[0])
        return scarlet_flux

    def kron_flux(self, im, noisy=True):
        back_i = sep.Background(im[self.indx])
        if noisy:
            img_sub = im[self.indx] - back_i
        else:
            img_sub = im[self.indx]

        cata, seg = sep.extract(
                img_sub, 1.5, err=back_i.globalrms, segmentation_map=True
        )


        kronrad, kronflag = sep.kron_radius(img_sub, cata['x'], cata['y'], cata['a'],
                                    cata['b'], cata['theta'], r=6.0)

        fluxes = np.zeros(len(im))
        fluxerrs = np.zeros_like(fluxes)
        for i,img in enumerate(im):
            if noisy:
                rms = back_i.globalrms
            else:
                rms = 0
            flux, fluxerr, flag = sep.sum_ellipse(img, cata['x'], cata['y'], cata['a'],
                                          cata['b'], cata['theta'], r = (2.5*kronrad),
                                          err = rms)
            if not noisy:
                fluxerr = np.sqrt(flux)
            fluxes[i] = flux
            fluxerrs[i] = fluxerr

        return fluxes, fluxerrs



    def scarlet_getsources(self, image, centers, psfs, indx=3):
        model_psf = scarlet.GaussianPSF(sigma=(.8,)*len(image))
        model_frame = scarlet.Frame(
            image.shape,
            psf=model_psf,
            channels=self.bands)
        rms_s = np.zeros(len(image))
        img_sub = np.zeros_like(image)
        for i,im in enumerate(image):
            bkg = sep.Background(im)
            img_sub[i] = im - bkg
            rms_s[i] = bkg.globalrms

        observation = scarlet.Observation(
            img_sub,
            psf=scarlet.ImagePSF(psfs),
            weights=np.ones(psfs.shape)/ (rms_s**2)[:,None,None],
            channels=self.bands).match(model_frame)

        # weights=np.ones(psfs.shape)/ (rms_s[indx]**2),
        sources, skipped = scarlet.initialization.init_all_sources(model_frame,
                                                               centers,
                                                               observation,
                                                               max_components=2,
                                                               min_snr=5,
                                                               thresh=1,
                                                               fallback=True,
                                                               silent=True,
                                                               set_spectra=True
                                                              )

    #     scarlet.initialization.set_spectra_to_match(sources, observation)
        blend = scarlet.Blend(sources, observation)
        it, logL = blend.fit(100, e_rel=1e-4)
        # Compute model
        model = blend.get_model()
        # Render it in the observed frame
        model_ = observation.render(model)
        # Compute residual
        residual = img_sub-model_
        # return sources, residual, model, observation
        return sources, residual, model, model_

    def get_surface(self):
        ellipse = np.pi * self.cata['a'][0] * self.cata['b'][0]
        return ellipse

    def pipeline(self, im, psfs):
        all_centers = np.array([[59,59]])
        sc, res, mod, obs = self.scarlet_getsources(im, all_centers, psfs)
        sf = self.comp_flux(sc)
        # kflux = self.kron_flux(mod)
        return sf, obs

    def compute_flux(self):
        sf_noisy, obs = self.pipeline(self.noisy, self.psfs)
        return sf_noisy, obs

    def lsst_mags_adu(self, magnitudes, filters, svy):
        conv_mag = np.zeros_like(magnitudes)
        for i,af in enumerate(filters):
            conv_mag[i] = mag2counts(magnitudes[i], svy, af).value
        return conv_mag

    def get_deblend(self):
        noisy_flux, obs = self.compute_flux()
        area = self.get_surface()

        return noisy_flux, area, obs


class MultiDeblend(SingleDeblend):

    def get_surface(self):
        ellipse = np.pi * self.cata['a'] * self.cata['b']
        return ellipse

    def pipeline(self, im, psfs):
        all_centers = self.get_center()
        sc, res, mod, obs = self.scarlet_getsources(im, all_centers, psfs)
        sf = self.comp_flux(sc)
        return sf, obs

    def comp_flux(self, sources):
        scarlet_flux = [scarlet.measure.flux(sc) for sc in sources]
        scarlet_flux = np.array(scarlet_flux)
        return scarlet_flux

    def compute_flux(self):
        sf_noisy = self.pipeline(self.noisy, self.psfs)
        return sf_noisy

    def get_deblend(self):
        noisy_flux = self.compute_flux()
        area = self.get_surface()

        return noisy_flux, area

    def get_center(self):
        cata = self.cata
        all_coords = [(cata['y'][i], cata['x'][i]) for i in range(len(cata['x']))]
        all_coords = np.array(all_coords)

        if len(all_coords) > 2:
            remove(["We've got more than 2 detections"])

        coord_norm = np.linalg.norm(all_coords - np.array([59,59]), axis=1)
        if np.argmin(coord_norm)==0:
            return all_coords
        else:
            new_coords = np.zeros_like(all_coords)
            new_coords[0,:] = all_coords[1,:]
            new_coords[1,:] = all_coords[0,:]
            
            return new_coords

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




def compare_flux(solved, true):
    N = len(solved)
    f = len(solved[0])
    ratios = np.zeros(f)
    err = np.zeros(f)

    for i in range(f):
        ratios[i] = np.mean([solv[i] for solv in solved])/true[0][i]
        err[i] = np.std([solv[i] for solv in solved])/(true[0][i] * np.sqrt(N))
    
    return ratios, err

def main_single(df, shift=4, saveFiles=True, verbose=False):
    catalog_name = "../data/sample_input_catalog.fits"
    stamp_size = 24
    bsize = 10
    surveys = btk.survey.get_surveys("LSST")
    bands = list('ugrizy')

    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    # sampling_function = btk.sampling_functions.DefaultSampling(max_number=5)
    sampling_function = CenteredSampling(setndx=shift)
    draw_blend_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        surveys,
        stamp_size=stamp_size,
        batch_size=bsize,
        cpus=1,
        add_noise='all'
    )

    if verbose:
        print("Created blend generator")

    batch = next(draw_blend_generator)
    blend_images = batch['blend_images']
    blend_list = batch['blend_list']
    blend_iso = batch['isolated_images']

    suvy = galcheat.get_survey("LSST")
    psfs = get_psfs(surveys, (120, 120))

    true_mags = np.array([catalog.table[shift][ab] for ab in ['u_ab', 'g_ab', 'r_ab', 'i_ab', 'z_ab', 'y_ab']])

    for bs in range(bsize):
        noisy_im = blend_images[bs]
        noiseless_im = blend_iso[bs][0]
        sblend = SingleDeblend(noisy_im, psfs)
        true_flux = sblend.lsst_mags_adu(true_mags, bands, suvy)
        res = sblend.get_deblend()
        # print(res, true_flux)

        df["cata_ndx"].append(shift)
        df["true_flux"].append(true_flux)
        df["noisy_flux"].append(res[0])
        df["surface_brightness"].append(true_flux[3]/res[1])

        df["noiseless_image"].append(noiseless_im)
        df["noisy_image"].append(noisy_im)
        df["reconstruction"].append(res[2])

    if verbose:
        print("Got fluxes")

    return 1

def main_double(df, shift=4, saveFiles=True, verbose=False):
    catalog_name = "../data/sample_input_catalog.fits"
    stamp_size = 24
    bsize = 10
    surveys = btk.survey.get_surveys("LSST")
    bands = list('ugrizy')

    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    # sampling_function = btk.sampling_functions.DefaultSampling(max_number=5)
    sampling_function = PairSampling(ndx=shift)
    draw_blend_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        surveys,
        stamp_size=stamp_size,
        batch_size=bsize,
        cpus=1,
        add_noise='all'
    )

    if verbose:
        print("Created blend generator")

    batch = next(draw_blend_generator)
    blend_images = batch['blend_images']
    blend_list = batch['blend_list']
    blend_iso = batch['isolated_images']

    suvy = galcheat.get_survey("LSST")
    psfs = get_psfs(surveys, (120, 120))

    true_mags = np.array([catalog.table[shift][ab] for ab in ['u_ab', 'g_ab', 'r_ab', 'i_ab', 'z_ab', 'y_ab']])

    for bs in range(bsize):
        noisy_im = blend_images[bs]
        noiseless_im = blend_iso[bs][0]
        sblend = MultiDeblend(noisy_im, psfs)
        true_flux = sblend.lsst_mags_adu(true_mags, bands, suvy)
        res = sblend.get_deblend()
        # print(res, true_flux)

        df["cata_ndx"].append(shift)
        df["true_flux"].append(true_flux)
        df["noisy_flux"].append(res[0])
        df["surface_brightness"].append(true_flux[3]/res[1])

        df["noiseless_image"].append(noiseless_im)
        df["noisy_image"].append(noisy_im)

    if verbose:
        print("Got fluxes")

    return 1
if __name__ == "__main__":
    # allshifts = np.array([ 0,  6, 21, 24, 36, 38, 43, 54, 61, 76, 87, 96, 98])
    allshifts = np.array([ 0,  6, 21, 24, 36, 38])
    # allshifts = np.array([88,75,56,41,49,77])
    lalls = len(allshifts)
    all_ratios = np.zeros((lalls, 6))
    all_errs = np.zeros_like(all_ratios)
    all_trues = np.zeros_like(all_ratios)
    noiseless_imgs = np.zeros((lalls, 6, 120, 120))
    noisy_imgs = []

        #   (Catalog Index, True Fluxes, Scarlet Fluxes, Ratio, Noiseless Image, Noisy Image)
    dframe_keys = ["cata_ndx", "true_flux", "noisy_flux", "noisy_image", "noiseless_image", "surface_brightness", "reconstruction"]
    dframe_solo = {sk:[] for sk in dframe_keys}
    dframe_double = {sk:[] for sk in dframe_keys}
    for i, sft in enumerate(allshifts):
        print(f"Processing index {i} object number {sft}")
        kale = main_single(dframe_solo, sft)
        print("Done with main single")
        # cabbage = main_double(dframe_double, sft)

    # remove(all_trues)
    # remove("sys exit now")
    prefix = "testSNR5_bkgsub_spectra_"
    fullframe_solo = pd.DataFrame(data=dframe_solo)
    fullframe_solo.to_pickle(f'../output/{prefix}pandaframe_solo.pkl')

    # fullframe_double = pd.DataFrame(data=dframe_double)
    # fullframe_double.to_pickle(f'../output/{prefix}pandaframe_double.pkl')

