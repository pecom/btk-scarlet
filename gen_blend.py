# -*- coding: utf-8 -*-
# Author : Prakruth Adari
# Email : prakruth.adari@stonybrook.edu

import astropy
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.table import QTable
import galsim
import galcheat

import numpy as np


import btk
import btk.plot_utils
import btk.survey
import btk.draw_blends
import btk.catalog
import btk.sampling_functions

from galcheat.utilities import mag2counts

# from sklearn.cluster import KMeans
from argparse import ArgumentParser
import pickle
import sys

class CenteredSampling(btk.sampling_functions.SamplingFunction):
    """ BTK function to place a single galaxy at the center of the post stamp """
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
    """
    BTK function to place a galaxy at the center of the postage stamp
    and another nearby (depends on self.setshift).
    
    Note this currently uses the same galaxy at both locations.
    """
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

def main_single(df, ndx=4, verbose=False, scale_size=1, catalog_name=None):
    """
    Create mock 'blends' with one galaxy pasted at the center 
    of the postage stamp.

    df - should be a dictionary that gets modified directly in the function (dictionary object)
    ndx - Catalog index of galaxy that will be pasted (integer)
    verbose - Include extra print statements or not (bool)
    scale_size - Increase size of galaxy parameters (float)
    """
    if catalog_name is None:
        catalog_name = "../catalogs/sample_input_catalog.fits"
    stamp_size = 24
    bsize = 10
    surveys = btk.survey.get_surveys("LSST")
    bands = list('ugrizy')

    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    catalog.table['a_d'] *= scale_size
    catalog.table['b_d'] *= scale_size
    catalog.table['a_b'] *= scale_size
    catalog.table['b_b'] *= scale_size
    # sampling_function = btk.sampling_functions.DefaultSampling(max_number=5)
    sampling_function = CenteredSampling(setndx=ndx)
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
        print("Created blend generator...")

    batch = next(draw_blend_generator)
    if verbose:
        print("Created blends...")
    blend_images = batch['blend_images']
    assert(blend_images[0][0].shape==(120,120))
    blend_list = batch['blend_list']
    blend_iso = batch['isolated_images']

    for bs in range(bsize):
        noisy_im = blend_images[bs]
        noiseless_im = blend_iso[bs]
        blend_cat = blend_list[bs]


        df["cata_ndx"].append(ndx)
        df["noiseless_image"].append(noiseless_im)
        df["noisy_image"].append(noisy_im)
        df["true_x"].append(blend_cat['x_peak'])
        df["true_y"].append(blend_cat['y_peak'])

    if verbose:
        print("Appended data ...")
 
    print(f"Done with galaxy {ndx}")
