## Python "transformers" code for comparison

Summary: I compared total calculation time for embeddings on the 21 strings in `../sentence_ex.tx` between HF transformers and transformers.f90 on an (old) Macbook and slightly newer intel/linux laptop. This code is heavily dependent on `matmul` so we expect the linear algebra backend to be important. On the intel machine, the inference runs 4-5 times faster in python vs the current fortran implementation with openblas. On the mac, the Accelerate framework + fortran is neatly 2x faster than python. Python generally sucks at the initial load time of the weights vs. fortran. 


Comparison on linux, intel core i7

```bash
$ time python transformer.py 
load time: 3.760972738265991
21
inferenece time: 0.37131738662719727

real	0m6.031s
user	0m5.267s
sys	0m2.265s
```

Compiled with O3 etc.

```bash
$ gfortran-10 -O3 -march=native -ffast-math -funroll-loops transformer.f90 -o tx
$ time ./tx -m msmarco-distilbert-base-dot-prod-v3.bin -q -f sentence_ex.txt --time
 Load time in seconds:    9.60000008E-02
 Total inference time in seconds:    4.36800003    

real	0m4.476s
user	0m4.376s
sys	0m0.100s
```

With external blas library

```bash
$ gfortran-10 -O3 -march=native -ffast-math -funroll-loops transformer.f90 -fexternal-blas -lopenblas -o tx
$ time ./tx -m msmarco-distilbert-base-dot-prod-v3.bin -q -f sentence_ex.txt --time
 Load time in seconds:   0.128000006    
 Total inference time in seconds:    1.28000009    

real	0m1.416s
user	0m9.457s
sys	0m7.029s
```

MacOS (intel, 2017 MBP)

```bash
% python transformer.py
load time: 0.9851226806640625
21
inferenece time: 2.890425205230713
```

```bash
% gfortran -O3 -march=native -ffast-math -funroll-loops transformer.f90 -o tx
% time ./tx -m msmarco-distilbert-base-dot-prod-v3.bin -q -f sentence_ex.txt --time
 Load time in seconds:   0.224000007
 Total inference time in seconds:    7.64800024
./tx -m msmarco-distilbert-base-dot-prod-v3.bin -q -f sentence_ex.txt --time  7.27s user 0.25s system 94% cpu 7.976 total
```

```bash
% gfortran -O3 -march=native -ffast-math -funroll-loops transformer.f90 -fexternal-blas -framework Accelerate -o tx
% time ./tx -m msmarco-distilbert-base-dot-prod-v3.bin -q -f sentence_ex.txt --time
 Load time in seconds:   0.352000028
 Total inference time in seconds:    1.50400007
./tx -m msmarco-distilbert-base-dot-prod-v3.bin -q -f sentence_ex.txt --time  1.47s user 0.19s system 76% cpu 2.178 total
```
