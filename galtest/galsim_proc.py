#! /home/padari/python/bin/python3.8

import numpy as np
import astropy.io.fits as fits
import os

dnums = 100

for p in range(dnums):

	# sys_str_noiseless = f"sex fits/demo1_{p}.fits -c noiseless.se -CATALOG_NAME cats/recon1_{p}.fits"
	# sys_str_noisy = f"sex fits/noisy1_{p}.fits -c noisy.se -CATALOG_NAME cats/noisy1_{p}.fits"

	sys_str_noiseless = f"sex fits/demo1_{p}.fits -c noiseless_radius.se -CATALOG_NAME cats/recon1_{p}.fits"
	sys_str_noisy = f"sex fits/noisy1_{p}.fits -c noisy_radius.se -CATALOG_NAME cats/noisy1_{p}.fits"
	os.system(sys_str_noiseless)
	os.system(sys_str_noisy)
