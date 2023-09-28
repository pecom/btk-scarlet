from abc import ABC, abstractmethod

import astropy.table
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Optional, Union

import btk
import btk.survey
import btk.draw_blends
import btk.catalog
import btk.sampling_functions

import scarlet

class CenteredSampling(btk.sampling_functions.SamplingFunction):

    def __init__(
        self,
        max_number=1,
        min_number=1,
        stamp_size=24.0,
        max_shift=0,
        ndx=0,
        seed=123
    ):
        super().__init__(max_number=max_number, min_number=min_number, seed=seed)
        self.stamp_size = stamp_size
        self.max_shift = max_shift if max_shift is not None else self.stamp_size / 10.0
        self.ndx = ndx

    @property
    def compatible_catalogs(self):
        return "CatsimCatalog", "CosmosCatalog"

    def __call__(self, table, shifts=None):
        blend_table = table[[self.ndx]]
        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0

        out_of_bounds = np.any(blend_table["ra"] > self.stamp_size / 2.0)
        out_of_bounds = out_of_bounds or np.any(blend_table["dec"] > self.stamp_size / 2.0)
        if out_of_bounds:
            warnings.warn("Object center lies outside the stamp")
        return blend_table


catalog_name = "../../data/input_catalog.fits"
catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
LSST = btk.survey.get_surveys("LSST")
allndx = np.array([76, 61, 54, 24, 0, 6])
allndx = np.array([76])
batch_size = 10
stamp_size = 24.0  # Size of the stamp, in arcseconds


for ndx in allndx:
    sampling_function = CenteredSampling(ndx=ndx)
    
    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        LSST,
        batch_size=batch_size,
        stamp_size=stamp_size,
        njobs=1,
        add_noise="all",
    )
    
    # generate batch 100 blend catalogs and images.
    blend_batch = next(draw_generator)
    
    deblender = btk.deblend.Scarlet(max_n_sources=1, max_iter=1000)
    deblend_batch = deblender(blend_batch, njobs=1)

    
    full_flux = np.einsum('ijk->i', blend_batch.isolated_images[0][0])
    realization_flux = np.einsum('abcde->ac', deblend_batch.deblended_images)
    # avg_realization_flux = np.mean(realization_flux, axis=0)
    scarlet_flux = np.array([np.array(list(scarlet.measure.flux(s['scarlet_sources'][0]).data)) 
                         for s in deblend_batch.extra_data])
    boot_err = np.array([np.std(stats.bootstrap([rf], np.mean).bootstrap_distribution)
                      for rf in realization_flux.T])
    fname_prefix = f'galaxy{ndx}_'
    np.save(f'../../output/btk_tests/{fname_prefix}truth', full_flux)
    np.save(f'../../output/btk_tests/{fname_prefix}realizations', realization_flux)
    np.save(f'../../output/btk_tests/{fname_prefix}scarlet', scarlet_flux)
    np.save(f'../../output/btk_tests/{fname_prefix}err', boot_err)
