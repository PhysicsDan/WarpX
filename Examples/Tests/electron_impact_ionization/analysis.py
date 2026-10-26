#!/usr/bin/env python3
#
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This script checks the results of the test `inputs_test_1d_electron_impact_ionization`,
# in which a uniform beam of 1 keV electrons ionizes a much less dense gas of carbon atoms.
# It checks that:
# - the populations of the charge states of carbon agree with the solution of the
#   ionization rate equations, using the Lotz cross sections implemented in WarpX,
# - the charge of the ejected electrons balances the charge gained by the ions,
# - the kinetic energy lost by the particles is equal to the energy spent on ionization,
# - the kinetic energy of the ejected electrons agrees with the binary-encounter model,
# - the total momentum of the particles is conserved.

import glob
import os

import numpy as np
import yt
from scipy import constants
from scipy.linalg import expm

yt.funcs.mylog.setLevel(0)

# Parameters of the simulation (see the input script)
n_e = 1.0e28  # beam density [m^-3]
E_beam = 1.0e3  # beam kinetic energy [eV]
dt = 5.0e-17  # time step [s]

# Ionization energies of carbon [eV] (see Source/Utils/Physics/IonizationEnergiesTable.H)
ionization_energies = np.array(
    [11.2602880, 24.383154, 47.88778, 64.49352, 392.090515, 489.993194]
)
Z = len(ionization_energies)

# Lotz cross section of each charge state of carbon, as computed in WarpX.
# The ground-state configuration of each charge state is obtained by filling the subshells
# in the order of the Madelung rule (only 1s, 2s and 2p are needed for carbon), and the binding
# energy of a subshell is the ionization energy of the charge state in which it is the
# outermost occupied subshell.
# Lotz's formula is extended to relativistic electrons by replacing E by m_e c^2 beta^2/2 in
# the denominator, and by adding a relativistic correction delta to the logarithmic term.
subshell_capacities = [2, 2, 6]
m_e_eV = constants.m_e * constants.c**2 / constants.e
gamma = 1.0 + E_beam / m_e_eV
beta_sq = 1.0 - 1.0 / gamma**2
E_beta = 0.5 * E_beam * (gamma + 1.0) / gamma**2
delta = np.log(0.5 * (gamma + 1.0)) - beta_sq
sigma = np.zeros(Z)
for charge_state in range(Z):
    n_bound = Z - charge_state
    n_filled = 0
    for capacity in subshell_capacities:
        if n_filled == n_bound:
            break
        occupation = min(capacity, n_bound - n_filled)
        n_filled += occupation
        binding_energy = ionization_energies[Z - n_filled]
        if E_beam > binding_energy:
            log_term = (
                np.log(E_beam / binding_energy)
                + (1.0 - (binding_energy / E_beam) ** 2) * delta
            )
            sigma[charge_state] += (
                4.5e-18 * occupation * log_term / (binding_energy * E_beta)
            )
print(f"Ionization cross sections [m^2]: {sigma}")

# Ionization rate of each charge state [s^-1]
v_beam = constants.c * np.sqrt(beta_sq)
rates = n_e * sigma * v_beam
print(f"Ionization rates [s^-1]: {rates}")

# Rate equations dN/dt = A N, with N[j] the fraction of ions in charge state j
A = np.zeros((Z + 1, Z + 1))
for j in range(Z):
    A[j, j] -= rates[j]
    A[j + 1, j] += rates[j]

# In WarpX, an ion can be ionized at most once per time step, with probability
# 1 - exp(-rate*dt). The corresponding discrete-time evolution, N^(n+1) = P N^n,
# differs from the solution of the rate equations by terms of order rate*dt.
P = np.eye(Z + 1)
for j in range(Z):
    probability = -np.expm1(-rates[j] * dt)
    P[j, j] -= probability
    P[j + 1, j] += probability

N_initial = np.zeros(Z + 1)
N_initial[0] = 1.0

# Compare the populations of the charge states of carbon with the theory
plotfiles = sorted(glob.glob(os.path.join("diags", "diag1" + "[0-9]" * 6)))
for plotfile in plotfiles:
    step = int(plotfile[-6:])
    if step == 0:
        continue
    ad = yt.load(plotfile).all_data()
    ion_level = ad["carbon", "particle_ionizationLevel"].v.astype(int)
    ion_weight = ad["carbon", "particle_weight"].v
    N_simulation = np.array(
        [ion_weight[ion_level == j].sum() for j in range(Z + 1)]
    ) / np.sum(ion_weight)
    N_discrete = np.linalg.matrix_power(P, step) @ N_initial
    N_continuous = expm(A * step * dt) @ N_initial

    # Statistical uncertainty on the fraction of ions in each charge state
    # (all the carbon macroparticles have the same weight in this test)
    N_uncertainty = np.sqrt(N_discrete * (1.0 - N_discrete) / ion_level.size)

    print(f"\nStep {step}:")
    print(f"  simulated charge state fractions:   {N_simulation}")
    print(f"  discrete-time model:                {N_discrete}")
    print(f"  solution of the rate equations:     {N_continuous}")
    assert np.all(np.abs(N_simulation - N_discrete) < 5.0 * N_uncertainty + 1.0e-3)

# Read the reduced diagnostics, using the column names from their headers
reduced_diags = {}
for name in ["particle_energy", "particle_momentum", "particle_number"]:
    filename = os.path.join("diags", "reducedfiles", name + ".txt")
    with open(filename) as f:
        header = f.readline().split()
    data = np.loadtxt(filename)
    reduced_diags[name] = {
        column.split("]")[1].split("(")[0]: data[:, i]
        for i, column in enumerate(header)
    }
energy = reduced_diags["particle_energy"]
momentum = reduced_diags["particle_momentum"]
number = reduced_diags["particle_number"]

# The final ionization levels and weights of the carbon ions are those of the last plotfile
assert int(plotfiles[-1][-6:]) == int(number["step"][-1])

# Charge conservation: each ionization event creates one ejected electron
ejected_weight = number["ejected_electrons_weight"][-1]
expected_ejected_weight = np.sum(ion_weight * ion_level)
print(f"\nWeight of the ejected electrons: {ejected_weight}")
print(f"Expected from the ionization levels: {expected_ejected_weight}")
assert np.isclose(ejected_weight, expected_ejected_weight, rtol=1.0e-12)

# The total weight of each colliding species is conserved
for species in ["electrons", "carbon"]:
    weight = number[species + "_weight"]
    assert np.allclose(weight, weight[0], rtol=1.0e-12)

# Energy conservation: the kinetic energy lost is the energy spent on ionization.
# (In the lab frame, the energy is conserved up to the kinetic energy of the ejected
# electrons when they were bound to the moving ions, which is negligible here.)
cumulative_ionization_energies = np.concatenate(([0.0], np.cumsum(ionization_energies)))
ionization_energy_spent = (
    np.sum(ion_weight * cumulative_ionization_energies[ion_level]) * constants.e
)
kinetic_energy_lost = energy["total"][0] - energy["total"][-1]
print(f"\nKinetic energy lost: {kinetic_energy_lost} [J]")
print(f"Energy spent on ionization: {ionization_energy_spent} [J]")
assert np.isclose(kinetic_energy_lost, ionization_energy_spent, rtol=1.0e-6)

# Kinetic energy of the ejected electrons. For each ionization event, the energy W = eps + I
# transferred from the incident electron (with eps the energy of the ejected electron) is
# sampled from dP/dW ~ 1/W^2 between I and (E_beam + I)/2. The number of ionization events
# from charge state j is given by the number of ions with an ionization level larger than j.
expected_ejected_energy = 0.0
ejected_energy_variance = 0.0
for j in range(Z):
    I_j = ionization_energies[j]
    W_max = 0.5 * (E_beam + I_j)
    normalization = 1.0 / (1.0 / I_j - 1.0 / W_max)
    W_mean = normalization * np.log(W_max / I_j)
    W_variance = normalization * (W_max - I_j) - W_mean**2
    n_events = np.sum(ion_level > j)
    expected_ejected_energy += n_events * (W_mean - I_j)
    ejected_energy_variance += n_events * W_variance
# All the ejected electrons have the same weight as the carbon macroparticles
ejected_energy = energy["ejected_electrons"][-1] / (ion_weight[0] * constants.e)
ejected_energy_uncertainty = np.sqrt(ejected_energy_variance)
print(f"\nKinetic energy of the ejected electrons: {ejected_energy} [eV]")
print(
    f"Expected: {expected_ejected_energy} +/- {ejected_energy_uncertainty} [eV] (1 sigma)"
)
assert abs(ejected_energy - expected_ejected_energy) < 5.0 * ejected_energy_uncertainty

# Momentum conservation. The momentum is conserved exactly in the rest frame of the ions.
# In the lab frame, it is conserved up to the momentum of the ejected electrons when they
# were bound to the moving ions (since the mass of the ions does not change when they are
# ionized), which is of order m_e/m_C times the momentum gained by the ions.
total_momentum = np.array(
    [momentum["total_x"], momentum["total_y"], momentum["total_z"]]
)
momentum_change = np.linalg.norm(total_momentum[:, -1] - total_momentum[:, 0])
relative_momentum_change = momentum_change / np.linalg.norm(total_momentum[:, 0])
print(f"\nRelative change of the total momentum: {relative_momentum_change}")
assert relative_momentum_change < 1.0e-7
