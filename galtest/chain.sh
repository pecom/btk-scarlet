# for i in {1..30}

# for j in 50 10 5 

# for j in 5 10 20
# for k in 1 2 5

# for j in 5 10 20
# do
# 	for i in 1 2 5 10 20 30
# 	do
# 		./galsim_images.py -noise $i -psf $j -size 2
# 		./galsim_proc.py
# 		./galsim_finform.py -noise $i -psf $j -size 2
# 	done
# done

# for k in 1 2 3
# do
# 	for i in 1 2 5 10 20 30
# 	do
# 		./galsim_images.py -noise $i -psf 3 -size $k
# 		./galsim_proc.py
# 		./galsim_finform.py -noise $i -psf 3 -size $k
# 	done
# done


for i in 1 2 5 10 20 30
do
	./galsim_images.py -noise $i -psf 4 -size .64
	./galsim_proc.py
	./galsim_finform.py -noise $i -psf 4 -size .64
done
