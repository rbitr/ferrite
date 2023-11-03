Python "transformers" code for comparison

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
