# btk-scarlet
NOTE: On jamaica we are just using python3 not python or any virtual environment

1. Run test/BTK_check.py to generate a set of FITS object
	- Generates sample images with BTK + Input Catalog
	- Detects with SEP
	- Deblends with Scarlet
	- Create data frame with all relevant images (original, isolated, reconstructed)
	- Stores Scarlet fluxes results
2. Run process_fits.py to get SExtractor output
	- Creates fits for each noisy / reconstructed image
	- Runs SEx on noisy to get background rms
	- Runs SEx on reconstructed to get relevant photometry. Uses forced photometry (detection in i-band, measurement in others)
3. Run fitspandas.py to get final data frame with just photometry
	- Formats all noisy/recon photometry into data frame
	- Properly estimates reconstructed errors for photometry
	- TODO: Make the scaling nmax read from the fits folder and properly yoink the largest number
 
