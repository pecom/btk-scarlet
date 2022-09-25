#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Prakruth Adari
# Email : prakruth.adari@stonybrook.edu


import astropy
from astropy.coordinates import SkyCoord
from astropy import units
import galsim
import matplotlib
import galcheat

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
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

def lsst_mags_adu(magnitudes, filters):
    suvy = galcheat.get_survey("LSST")
    conv_mag = np.zeros_like(magnitudes)

    for i,af in enumerate(filters):
        conv_mag[i] = mag2counts(magnitudes[i],suvy, af).value

    return conv_mag

def comp_flux(iso, sources, blist):
    scarlet_flux = scarlet.measure.flux(sources[0])
    btk_flux = np.einsum('ijk->i', iso[0])
#    scarlet_flux = [scarlet.measure.flux(sc) for sc in sources]
#     for bl in range(len(blist)):
#         btk_flux[bl,:] = np.einsum('ijk->i', iso[bl])
    return scarlet_flux, btk_flux

def get_center(image, indx=3):
    im = image[indx]
    back_i = sep.Background(im)
    img_sub = im - back_i

    cata, seg = sep.extract(
            img_sub, 1.5, err=back_i.globalrms, segmentation_map=True
    )
    all_coords = [(cata['x'][i], cata['y'][i]) for i in range(len(cata['x']))]

    all_coords = np.array(all_coords)
    return all_coords, cata, seg

def scarlet_getsources(image, centers, psfs, bands, indx=3):
    model_psf = scarlet.GaussianPSF(sigma=(.8,)*len(image))
    model_frame = scarlet.Frame(
        image.shape,
        psf=model_psf,
        channels=bands)
    rms_s = np.zeros(len(image))
    img_sub = np.zeros_like(image)
    for i,im in enumerate(image):
        bkg = sep.Background(im)
        img_sub[i] = im - bkg
        rms_s[i] = bkg.globalrms

    observation = scarlet.Observation(
        image,
        psf=scarlet.ImagePSF(psfs),
        weights=np.ones(psfs.shape)/ (rms_s[indx]**2),
        channels=bands).match(model_frame)

    sources, skipped = scarlet.initialization.init_all_sources(model_frame,
                                                           centers,
                                                           observation,
                                                           max_components=2,
                                                           min_snr=50,
                                                           thresh=1,
                                                           fallback=True,
                                                           silent=True,
                                                           set_spectra=False
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
    return sources, residual, model, observation

def sep_process(image, all_coords, segmap):
    lac = len(all_coords)
    sep_measures = np.zeros((lac, 6))
    for ac in range(lac):
        for j,im in enumerate(image):
            sep_measures[ac,j] = im[segmap==(ac+1)].sum()

    return sep_measures


def pair_pipeline(im, blist, psfs, iso):
    bands = list('ugrizy')

    all_centers, _, sep_segmap = get_center(im)
    sep_measure = sep_process(im, all_centers, sep_segmap)
    sc, res, mod, obs = scarlet_getsources(im, all_centers, psfs, bands)
    sf, btk_f = comp_flux(iso, sc, blist)

    return sf, btk_f

def compute_flux(svy, batch, blen=10):
    psfs = []
    pixscale = svy.pixel_scale
    im_shape = batch["blend_images"][0][0].shape
    for fp in svy.filters:
        psfs.append(fp.psf.drawImage(galsim.Image(*im_shape), scale=pixscale).array)
    psfs = np.array(psfs)

    
    sf_fluxes = []
    btk_fluxes = []

    for bl in range(blen):
        sf, btk_f = pair_pipeline(batch['blend_images'][bl], batch['blend_list'][bl], psfs, batch['isolated_images'][bl])
        sf_fluxes.append(sf)
        btk_fluxes.append(btk_f)
    return sf_fluxes, btk_fluxes

def plot_fit(fdat, filename):
    bands = list('ugrizy')
    fig, ax = plt.subplots(1, figsize=(8,6))

    cmap = cm.viridis
    lstyles = ['.-', '--', '-']
    lbls = ["Scarlet", "SEP", "Truth"]
    alphs = [1,.7, 1]

    for i, data in enumerate(fdat):
        for j,dd in enumerate(data):
            if j==0:
                lb = lbls
            else:
                lb = ["","",""]
            ax.plot(dd, lstyles[i], color=cmap(i/3), alpha=alphs[i], label=lb[i])

    ax.set_ylabel("Magnitude", fontsize=14)
    ax.set_xlabel("Band", fontsize=14)
    ax.legend(frameon=False,fontsize=16,bbox_to_anchor=(1.01,.5), loc='center left')
    ax.set_ylim(top=29)
    # fig.legend(["SEP", "Scarlet", "Catalog"],frameon=False,fontsize=16,bbox_to_anchor=(1.01,.5), loc='center left')
    ax.set_xticks(np.arange(len(bands)), bands, fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_title("Magnitude after Deblending")
    plt.tight_layout()
    plt.savefig('../images/'+filename)
    # plt.show()

def plot_blend(bness, filename):
    bands = list('ugrizy')
    fig, ax = plt.subplots(1, figsize=(8,6))

    cmap = cm.viridis
    for bn in bness:
        ax.plot(bn, '.-')

    ax.set_ylim(bottom =0)
    ax.set_ylabel("Blendedness", fontsize=14)
    ax.set_xlabel("Band", fontsize=14)
    ax.set_xticks(np.arange(len(bands)), bands, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('../images/'+filename)
    # plt.show()


def get_blendedness(batch, bands):
    biso = batch['isolated_images']
    band_blends = np.zeros(biso.shape[:3])
    for i, bis in enumerate(biso):
        sall_bands = np.sum(bis, axis=0) 
        for j, bi in enumerate(bis):
            if np.sum(bi) == 0:
                band_blends[i,j,:] = -1
                continue
            for k, b in enumerate(bi):
                sall = sall_bands[k]
                beta = 1 - (b*b).sum()/(b*sall).sum()
                band_blends[i,j,k] = beta
    return band_blends


def compare_flux(solved, true):
    N = len(solved)
    f = len(solved[0])
    ratios = np.zeros(f)
    err = np.zeros(f)

    for i in range(f):
        ratios[i] = np.mean([solv[i] for solv in solved])/true[0][i]
        err[i] = np.std([solv[i] for solv in solved])/(true[0][i] * np.sqrt(N))
    
    return ratios, err

def main(shift=4, saveFiles=True, verbose=False):
    bands = list('ugrizy')
    catalog_name = "../data/blending/sample_input_catalog.fits"
    stamp_size = 24
    bsize = 5
    surveys = btk.survey.get_surveys("Rubin")

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

    # btk.plot_utils.plot_blends(blend_images, blend_list)
    
    true_flux = np.einsum('ijklm->ijk', blend_iso)
    if verbose:
        print("Created blend batch")
    blendedness = get_blendedness(batch, bands)
    if verbose:
        print("Calculated blendedness")
    sf_fluxes, btk_fluxes = compute_flux(surveys, batch, len(blend_list))
    if verbose:
        print("Got fluxes")

    true_mags = np.array([catalog.table[shift][ab] for ab in ['u_ab', 'g_ab', 'r_ab', 'i_ab', 'z_ab', 'y_ab']])
    # ab_zerop = np.array([f.zeropoint for f in surveys.filters])
    # zerop_expo = np.array([f.exp_time for f in surveys.filters])
    true_flux = lsst_mags_adu(true_mags, bands)
    multi_true = [true_flux for i in range(bsize)]
    ratio, errs = compare_flux(sf_fluxes, multi_true)

    if saveFiles:
        outdir = '../output/single/'
        with open(f'{outdir}scarlet_{shift}_flux.pkl', 'wb') as fn:
            pickle.dump(sf_fluxes, fn)
        with open(f'{outdir}btk_{shift}_flux.pkl', 'wb') as fn:
            pickle.dump(btk_fluxes, fn)
        np.save(f'{outdir}blendedness.npy', blendedness)
        np.save(f'{outdir}ratios.npy', ratio)
        np.save(f'{outdir}errs.npy', errs)

    # fdata = [sf_m[1], sep_m[1], cf_m[1]]
    # plot_fit(fdata, 'single_blendtest.jpeg')
    # plot_blend(blendedness[1], 'single_blendednesstest.jpeg')
    return ratio, errs

allshifts = np.array([ 0,  6, 21, 24, 36, 38, 43, 54, 61, 76, 87, 96, 98])
all_ratios = np.zeros((len(allshifts), 6))
all_errs = np.zeros_like(all_ratios)
for i, sft in enumerate(allshifts):
    print(f"Processing index {i} object number {sft}")
    r, e = main(sft)
    all_ratios[i,:] = r
    all_errs[i,:] = e

# sys.exit()
suffix = "nospectra"
np.save(f'../output/single/allratios_{suffix}.npy', all_ratios)
np.save(f'../output/single/allerrs_{suffix}.npy', all_errs)

