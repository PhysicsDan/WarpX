Building the spectral solver
============================

By default, the code is compiled with a finite-difference (FDTD) Maxwell solver.
In order to run the code with a spectral solver, you need to:

      - Install (or load) an MPI-enabled version of FFTW.
        For instance, for Debian, this can be done with
        ::

           apt-get install libfftw3-dev libfftw3-mpi-dev

      - Set the environment variable ``FFTW_HOME`` to the path for FFTW.
        For instance, for Debian, this is done with
        ::

           export FFTW_HOME=/usr/

      - Set ``USE_FFT=TRUE`` when compiling:
        ::

           make -j 4 USE_FFT=TRUE

See :doc:`rzgeometry` for using the spectral solver with USE_RZ. Additional steps are needed.
PSATD is compatible with single precision, but please note that, on CPU, FFTW needs to be compiled with option ``--enable-float``.
