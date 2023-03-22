for i in 2 5 10 20 30
do
	./galsim_images.py -noise $i
	./galsim_proc.py
	./galsim_finform.py -noise $i
done
