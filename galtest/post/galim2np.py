import numpy as np
import astropy.io.fits as fits

with fits.open(f'../fits/demo1_1.fits') as f:
	kale = f[0].data

sample_ims = np.zeros((20,*kale.shape))

for i in range(20):
	with fits.open(f'../fits/demo1_{i}.fits') as f:
		kale = f[0].data
	sample_ims[i,:,:] = kale

np.save('./output/scarlet_ims.npy', sample_ims)
