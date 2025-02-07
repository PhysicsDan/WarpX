#!/usr/bin/env python3

# Copyright 2019 Maxence Thevenet
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


import sys

import numpy as np
import yt
from scipy import signal

yt.funcs.mylog.setLevel(0)

# Build Jx without filter. This can be obtained by running this test without
# a filter, e.g., execute
# > OMP_NUM_THREADS=2 mpirun -np 2 ~/warpx/Bin/main2d.gnu.TPROF.MPI.OMP.ex \
#   inputs warpx.use_filter=0
# instead of
# > OMP_NUM_THREADS=2 mpirun -np 2 ~/warpx/Bin/main2d.gnu.TPROF.MPI.OMP.ex \
#   inputs warpx.use_filter=1 warpx.filter_npass_each_dir=1 5
# and then print the values in the array F_filtered below.
my_F_nofilter = np.zeros([16, 16])
my_F_nofilter[8, 8] = -1.601068237523421e-11
my_F_nofilter[8, 7] = -1.601068237523421e-11

# Build 2D filter
filter0 = np.array([0.25, 0.5, 0.25])
my_order = [1, 5]
my_filterx = filter0
my_filtery = filter0
while my_order[0] > 1:
    my_filterx = np.convolve(my_filterx, filter0)
    my_order[0] -= 1
while my_order[1] > 1:
    my_filtery = np.convolve(my_filtery, filter0)
    my_order[1] -= 1
my_filter = my_filterx[:, None] * my_filtery

# Apply filter. my_F_filtered is the theoretical value for filtered field
my_F_filtered = signal.convolve2d(
    my_F_nofilter, my_filter, boundary="symm", mode="same"
)

# Get simulation result for F_filtered
filename = sys.argv[1]
ds = yt.load(filename)
sl = yt.SlicePlot(ds, 2, "jx", aspect=1)
all_data_level_0 = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
F_filtered = all_data_level_0["boxlib", "jx"].v.squeeze()

# Compare theory and PIC for filtered value
error_rel = np.sum(np.abs(F_filtered - my_F_filtered)) / np.sum(np.abs(my_F_filtered))
tolerance_rel = 1.0e-14

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel
