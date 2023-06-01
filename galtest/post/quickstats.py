import numpy as np
import astropy.io.fits as fits
import sep

infty = np.zeros(100)
circle = np.zeros(100)


for i in range(100):
	with fits.open(f'./fits/demo1_{i}.fits') as f:
		kale = f[0].data.byteswap().newbyteorder()
	bkg = sep.Background(kale)
	corr = kale - bkg
	infty[i] = np.sum(corr)
	circle[i] = sep.sum_circle(corr, [66], [66], [22])[0][0]
	
np.save('./output/infty_bkgsub.npy', infty)
np.save('./output/circle_bkgsub.npy', circle)
