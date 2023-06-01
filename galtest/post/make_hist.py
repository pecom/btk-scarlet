import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('Agg')

import matplotlib.pyplot as plt


grecon = pd.read_pickle('../output/galsim_recon_frame.pkl')

grkf = grecon['kron_flux']
print(f"Mean error: {np.mean(grecon['kron_err'])}")
print(f"Min of kron flux: {np.min(grkf), np.argmin(grkf)}")
print(f"Min max of kron rad: {np.min(grecon['kron_rad']), np.max(grecon['kron_rad'])}")

clean_arr = np.concatenate((grkf[:63], grkf[64:]))
# clean_arr = grkf

plt.hist(clean_arr, range=(8e4, 1.2e5))
plt.axvline(1.e5, color='black', label='Input Flux')
plt.axvline(np.mean(clean_arr), color='black', label='Mean Reconstructed Flux', ls='--')
# xt = np.array([85000, 90000, 95000, 100000, 105000])

# plt.xticks(xt, xt)
# plt.axvline(np.mean(gnoisy['kron_flux']), color='red', label='SExtractor')
plt.legend(loc=(1.1, .45), fontsize=14, frameon=False)
plt.title("Kron Flux from 100 Realizations with Galsim w/ 1 Components", fontsize=14)
plt.savefig('../figs/galsim_kronrecon_n1_sigma10.png', dpi=100)
