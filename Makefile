FORTRAN = gfortran-10
GCC = gcc-10

.DEFAULT_GOAL := all

weight_module.o: weight_module.f90 
	$(FORTRAN) -c -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fPIC weight_module.f90

transformer.o: transformer.f90 
	$(FORTRAN) -c -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fPIC transformer.f90  
read_ggml.o: read_ggml.f90
	$(FORTRAN) -c -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fPIC read_ggml.f90

tx: weight_module.o read_ggml.o transformer.o 
	$(FORTRAN) -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fPIC weight_module.o read_ggml.o transformer.o -o tx 

	

all: tx

clean:
	rm *.o
