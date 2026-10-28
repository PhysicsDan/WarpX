#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# Three identical cold beams of 100 eV electrons go through carbon ions at rest, and their
# momentum decays by pairwise Coulomb collisions (see inputs_test_2d_collision_xz_ionizable).
# Since the ions are much heavier than the electrons, the momentum of each beam decays as
# p_z(t) = p_z(0) exp(-nu t), with the momentum-transfer rate
#   nu = n_i Z^2 e^4 lnL / (4 pi eps0^2 m_e^2 v^3)
# where the Coulomb logarithm is computed as in UpdateMomentumPerezElastic.H: since the beam is
# cold, the screening length is the atomic spacing, and the minimum impact parameter is the
# largest of the classical and quantum ones.
# We check that:
# - the momentum of the beams colliding with ions without field ionization (charge Z q_e) and
#   with ions with field ionization (ionization level Z) decays at the expected rate,
# - the momentum of the beam colliding with neutral atoms (ionization level 0) does not change.

import numpy as np
from scipy import constants

n_i = 1.0e26
Z = 6
E_beam = 100.0  # [eV]

c = constants.c
e = constants.e
m_e = constants.m_e
eps0 = constants.epsilon_0
hbar = constants.hbar

# Expected momentum-transfer rate
gamma = 1.0 + E_beam * e / (m_e * c**2)
v = c * np.sqrt(1.0 - 1.0 / gamma**2)
p = gamma * m_e * v
atomic_spacing = 1.0 / np.cbrt(4.0 * np.pi / 3.0 * Z * n_i)
b0 = Z * e**2 / (2.0 * np.pi * eps0 * m_e * v**2)
bmin = max(hbar / (2.0 * p), 0.5 * b0)
lnL = max(2.0, 0.5 * np.log(1.0 + atomic_spacing**2 / bmin**2))
nu_expected = n_i * Z**2 * e**4 * lnL / (4.0 * np.pi * eps0**2 * m_e**2 * v**3)
print(f"Expected rate: {nu_expected:.4e} 1/s (Coulomb logarithm {lnL:.3f})")

# Momentum of the three electron beams
path = "diags/reducedfiles/particle_momentum.txt"
header = open(path).readline().split()
columns = [name.split("]")[1] for name in header]
data = np.loadtxt(path, skiprows=1)
time = data[:, columns.index("time(s)")]

for species in ["electrons_ref", "electrons_ion", "electrons_neu"]:
    pz = data[:, columns.index(f"{species}_z(kg*m/s)")]
    if species == "electrons_neu":
        relative_change = np.max(np.abs(pz / pz[0] - 1.0))
        print(
            f"{species}: maximum relative change of the momentum {relative_change:.2e}"
        )
        assert relative_change < 1.0e-12
    else:
        nu = -np.polyfit(time, np.log(pz / pz[0]), 1)[0]
        print(
            f"{species}: rate {nu:.4e} 1/s, relative error {nu / nu_expected - 1.0:.4f}"
        )
        assert abs(nu / nu_expected - 1.0) < 0.05
