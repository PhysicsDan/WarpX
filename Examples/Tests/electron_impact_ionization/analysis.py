#!/usr/bin/env python3
#
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This script checks the results of the tests `inputs_test_1d_electron_impact_ionization` and
# `inputs_test_1d_electron_impact_ionization_split`, in which a uniform beam of electrons ionizes
# a gas of carbon atoms. It checks that:
# - the populations of the charge states of carbon agree with the solution of the ionization rate
#   equations, using the MBELL cross sections and the binding energies implemented in WarpX
#   (when the macroparticles are split, each ion is ionized at most once per time step, and the
#   corresponding discrete-time evolution is used),
# - the charge of the ejected electrons balances the charge gained by the ions,
# - the kinetic energy lost by the electrons and ions is equal to the binding energy of the ionized
#   subshells plus the kinetic energy of the ejected electrons (on average),
# - the kinetic energy of the ejected electrons agrees with the RBEB differential cross section,
# - at each output step, the total charge is conserved exactly, the total kinetic energy decreases
#   by the binding energy of the ionized subshells, and the total momentum is conserved: exactly
#   when the macroparticles are split, and within the statistical bounds of the default method
#   otherwise (in which energy and momentum are only conserved on average).

import glob
import os

import numpy as np
import yt
from scipy import constants
from scipy.integrate import trapezoid
from scipy.linalg import expm

yt.funcs.mylog.setLevel(0)

# Parameters of the simulation, read from the input file
inputs = {}
with open("warpx_used_inputs") as f:
    for line in f:
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            inputs[key.strip()] = value.split("#")[0].strip()
n_e = float(inputs["my_constants.n_e"])  # beam density [m^-3]
E_beam = float(inputs["my_constants.E_beam"])  # beam kinetic energy [eV]
dt = float(inputs["warpx.const_dt"])  # time step [s]
split_macroparticles = inputs.get("ionization.split_macroparticles", "0") in [
    "1",
    "true",
]
print(f"Macroparticles are split: {split_macroparticles}")

# Ionization energies of carbon [eV] (see Source/Utils/Physics/IonizationEnergiesTable.H) and
# binding energies of the electrons of the neutral carbon atom, from the most bound
# (see Source/Utils/Physics/binding_energy_data.txt)
ionization_energies = np.array(
    [11.2602880, 24.383154, 47.88778, 64.49352, 392.090515, 489.993194]
)
neutral_binding_energies = np.array([288.23, 288.23, 16.59, 16.59, 11.26, 11.26])
Z = len(ionization_energies)
m_e_eV = constants.m_e * constants.c**2 / constants.e

# Parameters of the MBELL model for the subshells 1s, 2s and 2p (Haque et al. 2006)
mbell_A = [0.525, 0.530, 0.600]
mbell_B = [
    [-0.510, 0.2000, 0.0500, -0.025, -0.100, 0.00, 0.00],
    [-0.410, 0.1500, 0.1500, -0.200, -0.150, 0.00, 0.00],
    [-0.400, -0.7100, 0.6550, 0.425, -0.750, 0.00, 0.00],
]
mbell_lambda = [1.27, 1.27, 0.542]

# RBEB parameters that only depend on the energy of the incident electron
tp = E_beam / m_e_eV
beta_t_sq = 1.0 - 1.0 / (1.0 + tp) ** 2

# For each charge state, compute the cross section of each occupied subshell for an incident
# electron of energy E_beam, and the moments of the binding energy B of the ionized subshell
# and of the kinetic energy W of the ejected electron (sampled from the RBEB singly
# differential cross section), which are needed to compute the expected energy changes.
sigma = np.zeros(Z)
mean_B = np.zeros(Z)
mean_W = np.zeros(Z)
var_B = np.zeros(Z)
var_W = np.zeros(Z)
var_BW = np.zeros(Z)  # variance of B + W
for q in range(Z):
    n_electrons = Z - q
    # The ion keeps the n_electrons most bound electrons of the neutral atom, which are grouped
    # into subshells 1s, 2s, 2p (each pair of electrons of carbon has the same binding energy),
    # and the binding energies are shifted such that the binding energy of the outermost
    # electron is the ionization energy of the ion (Carlson et al. 1970)
    binding = neutral_binding_energies[:n_electrons]
    shift = ionization_energies[q] - binding[-1]
    sigma_s, B_s, W_s, W2_s = [], [], [], []
    for subshell in range(3):
        N = np.sum(np.arange(n_electrons) // 2 == subshell)
        if N == 0:
            continue
        B = neutral_binding_energies[2 * subshell] + shift
        # Number of electrons in this subshell and in the more bound subshells
        q_eff = Z - np.sum(np.arange(n_electrons) // 2 <= subshell)
        u = E_beam / B
        J = m_e_eV / B
        bell = (
            mbell_A[subshell] * np.log(u)
            + sum(mbell_B[subshell][k] * (1.0 - 1.0 / u) ** (k + 1) for k in range(7))
        ) / (B * E_beam)
        gryzinski = (
            (1.0 + 2.0 * J)
            / (u + 2.0 * J)
            * ((u + J) / (1.0 + J)) ** 2
            * (
                (1.0 + u)
                * (u + 2.0 * J)
                * (1.0 + J) ** 2
                / (J**2 * (1.0 + 2.0 * J) + u * (u + 2.0 * J) * (1.0 + J) ** 2)
            )
            ** 1.5
        )
        f_ion = 1.0 + 3.0 * (q_eff / (Z * u)) ** mbell_lambda[subshell]
        sigma_s.append(N * f_ion * gryzinski * bell * 1.0e-17)  # [m^2]

        # Singly differential cross section of the RBEB model, as a function of w = W/B
        t = E_beam / B
        bp = B / m_e_eV
        A3 = np.log(beta_t_sq / (1.0 - beta_t_sq)) - beta_t_sq - np.log(2.0 * bp)
        C1 = (1.0 + 2.0 * tp) / (1.0 + 0.5 * tp) ** 2
        C2 = bp**2 / (1.0 + 0.5 * tp) ** 2
        w = np.linspace(0.0, 0.5 * (t - 1.0), 2000001)
        dsigma_dw = (
            1.0 / (w + 1.0) ** 2
            + 1.0 / (t - w) ** 2
            - C1 / (t + 1.0) * (1.0 / (w + 1.0) + 1.0 / (t - w))
            + C2
            + A3 * (1.0 / (w + 1.0) ** 3 + 1.0 / (t - w) ** 3)
        )
        normalization = trapezoid(dsigma_dw, w)
        B_s.append(B)
        W_s.append(B * trapezoid(w * dsigma_dw, w) / normalization)
        W2_s.append(B**2 * trapezoid(w**2 * dsigma_dw, w) / normalization)

    sigma_s, B_s, W_s, W2_s = map(np.array, [sigma_s, B_s, W_s, W2_s])
    sigma[q] = np.sum(sigma_s)
    probability = sigma_s / sigma[q]
    mean_B[q] = np.sum(probability * B_s)
    mean_W[q] = np.sum(probability * W_s)
    var_B[q] = np.sum(probability * B_s**2) - mean_B[q] ** 2
    var_W[q] = np.sum(probability * W2_s) - mean_W[q] ** 2
    var_BW[q] = (
        np.sum(probability * (B_s**2 + 2.0 * B_s * W_s + W2_s))
        - (mean_B[q] + mean_W[q]) ** 2
    )
print(f"Ionization cross sections [m^2]: {sigma}")
print(f"Mean binding energy of the ionized subshell [eV]: {mean_B}")
print(f"Mean kinetic energy of the ejected electrons [eV]: {mean_W}")

# Ionization rate of each charge state [s^-1]
v_beam = constants.c * np.sqrt(beta_t_sq)
rates = n_e * sigma * v_beam
print(f"Ionization rates [s^-1]: {rates}")

# Rate equations dN/dt = A N, with N[j] the fraction of ions in charge state j
A = np.zeros((Z + 1, Z + 1))
for j in range(Z):
    A[j, j] -= rates[j]
    A[j + 1, j] += rates[j]

# When the macroparticles are split, an ion can be ionized at most once per time step, with
# probability 1 - exp(-rate*dt). The corresponding discrete-time evolution, N^(n+1) = P N^n,
# differs from the solution of the rate equations by terms of order rate*dt.
P = np.eye(Z + 1)
for j in range(Z):
    probability = -np.expm1(-rates[j] * dt)
    P[j, j] -= probability
    P[j + 1, j] += probability

N_initial = np.zeros(Z + 1)
N_initial[0] = 1.0

# Compare the populations of the charge states of carbon with the theory, and record at each
# output step the number of ionization events from each charge state (number of ions with a
# larger ionization level) and the charge of the ions [e] (the carbon atoms are initially neutral)
plotfiles = sorted(glob.glob(os.path.join("diags", "diag1" + "[0-9]" * 6)))
events_at_step = {0: np.zeros(Z)}
ion_charge_at_step = {0: 0.0}
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
    N_theory = N_discrete if split_macroparticles else N_continuous

    # Statistical uncertainty on the fraction of ions in each charge state
    # (all the carbon macroparticles have the same weight in these tests)
    N_uncertainty = np.sqrt(N_theory * (1.0 - N_theory) / ion_level.size)

    print(f"\nStep {step}:")
    print(f"  simulated charge state fractions:   {N_simulation}")
    print(f"  discrete-time model:                {N_discrete}")
    print(f"  solution of the rate equations:     {N_continuous}")
    assert np.all(np.abs(N_simulation - N_theory) < 5.0 * N_uncertainty + 1.0e-3)
    events_at_step[step] = np.array([np.sum(ion_level > q) for q in range(Z)])
    ion_charge_at_step[step] = np.sum(ion_weight * ion_level)

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
w_i = ion_weight[0]
assert np.allclose(ion_weight, w_i, rtol=1.0e-12)
# Weight of the beam electrons (all the macroparticles of the beam initially have the same weight)
w_e = number["electrons_weight"][0] / number["electrons_macroparticles"][0]

# Charge conservation: each ionization event creates one ejected electron
ejected_weight = number["ejected_electrons_weight"][-1]
expected_ejected_weight = np.sum(ion_weight * ion_level)
print(f"\nWeight of the ejected electrons: {ejected_weight}")
print(f"Expected from the ionization levels: {expected_ejected_weight}")
assert np.isclose(ejected_weight, expected_ejected_weight, rtol=1.0e-12)
if not split_macroparticles:
    # Each ionization of an ion macroparticle creates one ejected electron macroparticle
    assert number["ejected_electrons_macroparticles"][-1] == np.sum(ion_level)

# The total weight of each colliding species is conserved
for species in ["electrons", "carbon"]:
    weight = number[species + "_weight"]
    assert np.allclose(weight, weight[0], rtol=1.0e-12)

# Number of ionization events from each charge state, given by the number of ions with an
# ionization level larger than this charge state
n_events = np.array([np.sum(ion_level > q) for q in range(Z)])
print(f"\nNumber of ionization events from each charge state: {n_events}")

# Kinetic energy of the ejected electrons: each ionization event creates an electron of weight
# w_i, whose kinetic energy is sampled from the RBEB differential cross section
ejected_energy = energy["ejected_electrons"][-1] / constants.e
expected_ejected_energy = w_i * np.sum(n_events * mean_W)
ejected_energy_uncertainty = w_i * np.sqrt(np.sum(n_events * var_W))
print(f"\nKinetic energy of the ejected electrons: {ejected_energy} [eV]")
print(
    f"Expected: {expected_ejected_energy} +/- {ejected_energy_uncertainty} [eV] (1 sigma)"
)
assert abs(ejected_energy - expected_ejected_energy) < 5.0 * ejected_energy_uncertainty

# Kinetic energy lost by the colliding species: in each ionization event, the incident
# electron loses the binding energy B of the ionized subshell plus the kinetic energy W of the
# ejected electron. When the macroparticles are split, the energy is conserved exactly for each
# event, of weight w_i. Otherwise, each event is applied to the electron macroparticle with the
# probability w_i/w_e (since w_e > w_i in the test), which conserves the energy on average.
energy_lost = (
    energy["electrons"][0]
    + energy["carbon"][0]
    - energy["electrons"][-1]
    - energy["carbon"][-1]
) / constants.e
expected_energy_lost = w_i * np.sum(n_events * (mean_B + mean_W))
if split_macroparticles:
    energy_lost_variance = w_i**2 * np.sum(n_events * var_BW)
else:
    mean_square_BW = var_BW + (mean_B + mean_W) ** 2
    energy_lost_variance = np.sum(
        n_events * (w_e * w_i * mean_square_BW - w_i**2 * (mean_B + mean_W) ** 2)
    )
energy_lost_uncertainty = np.sqrt(energy_lost_variance)
print(f"\nKinetic energy lost by the electrons and the ions: {energy_lost} [eV]")
print(f"Expected: {expected_energy_lost} +/- {energy_lost_uncertainty} [eV] (1 sigma)")
assert abs(energy_lost - expected_energy_lost) < 5.0 * energy_lost_uncertainty

# Conservation of the total charge, kinetic energy and momentum, at each output step:
# - charge: the charge gained by the ions is that of the ejected electrons (exactly);
# - energy: the total kinetic energy decreases by the binding energy B of the ionized subshell in
#   each event, of weight w_i. When the macroparticles are split, this holds exactly for each
#   event, but B is only known on average for each charge state (mean_B, var_B), which sets the
#   uncertainty. With the default method, the energy change -(B + W) of the incident electron is
#   applied to its macroparticle, of weight w_e, with the probability w_i/w_e, and the kinetic
#   energy W' of the ejected electron is sampled independently of W, when the electron is created
#   after the collisions: the energy change of an event, w_i W' - w_e X (B + W) (X = 1 with the
#   probability w_i/w_e, 0 otherwise), is -w_i B on average, with the variance
#   w_i^2 var(W) + w_e w_i <(B + W)^2> - w_i^2 <B + W>^2;
# - momentum: when the macroparticles are split, it is conserved exactly in the rest frame of the
#   ion of each event. In the lab frame, it is conserved up to the momentum of the ejected electrons
#   when they were bound to the moving ions (since the mass of the ions does not change when they
#   are ionized), which is of order m_e/m_C times the momentum gained by the ions through their
#   recoil. (Without the recoil of the ions, the relative change of the total momentum would be of
#   order 1e-3 in this test.) With the default method, the total momentum changes in each event by
#   w_i (p' - p) + (w_e X - w_i) d, where d is the momentum change of the incident electron, p the
#   momentum of the ejected electron used to compute d and the recoil of the ion, and p' that of the
#   ejected electron created (sampled independently): zero on average, with the variance
#   2 w_i^2 var(p) + (w_e w_i - w_i^2) d^2 for each component (checked at the last step).
total_momentum = np.array(
    [momentum["total_x"], momentum["total_y"], momentum["total_z"]]
)
initial_negative_charge = number["electrons_weight"][0]  # [e]
mean_square_BW = var_BW + (mean_B + mean_W) ** 2
print("\nConservation laws at each output step:")
for k, step in enumerate(number["step"].astype(int)):
    events = events_at_step[step]
    total_charge = (
        ion_charge_at_step[step]
        - number["electrons_weight"][k]
        - number["ejected_electrons_weight"][k]
    )
    charge_change = total_charge + initial_negative_charge
    energy_change = (energy["total"][k] - energy["total"][0]) / constants.e
    expected_energy_change = -w_i * np.sum(events * mean_B)
    if split_macroparticles:
        energy_variance = w_i**2 * np.sum(events * var_B)
    else:
        energy_variance = np.sum(
            events
            * (
                w_i**2 * var_W
                + w_e * w_i * mean_square_BW
                - w_i**2 * (mean_B + mean_W) ** 2
            )
        )
    energy_uncertainty = np.sqrt(energy_variance)
    relative_momentum_change = np.linalg.norm(
        total_momentum[:, k] - total_momentum[:, 0]
    ) / np.linalg.norm(total_momentum[:, 0])
    print(
        f"  step {step:3d}: charge change {charge_change:.3e} e, kinetic energy change "
        f"{energy_change:.6e} eV (expected {expected_energy_change:.6e} +/- "
        f"{energy_uncertainty:.2e}), relative momentum change {relative_momentum_change:.2e}"
    )
    assert abs(charge_change) < 1.0e-12 * initial_negative_charge
    assert abs(energy_change - expected_energy_change) <= 5.0 * energy_uncertainty + (
        1.0e-12 * energy["total"][0] / constants.e
    )
    if split_macroparticles:
        assert relative_momentum_change < 2.0e-7

if not split_macroparticles:
    # The sum of d^2 over the events is estimated from the momentum changes of the beam electrons
    # at the last step, whose sum of squares is (w_i/w_e) sum(d^2) (each event being applied with
    # the probability w_i/w_e), and the sum of var(p) from the momenta of the ejected electrons
    step = int(number["step"][-1])
    ad = yt.load(os.path.join("diags", f"diag2{step:06d}")).all_data()
    p_beam_initial = np.array(
        [0.0, 0.0, constants.m_e * constants.c * np.sqrt((1.0 + tp) ** 2 - 1.0)]
    )
    sum_square_beam = np.array(
        [
            np.sum(
                (ad["electrons", f"particle_momentum_{d}"].v - p_beam_initial[i]) ** 2
            )
            for i, d in enumerate("xyz")
        ]
    )
    sum_square_ejected = np.array(
        [
            np.sum(
                (
                    ad["ejected_electrons", f"particle_momentum_{d}"].v
                    - np.mean(ad["ejected_electrons", f"particle_momentum_{d}"].v)
                )
                ** 2
            )
            for d in "xyz"
        ]
    )
    momentum_uncertainty = np.sqrt(
        w_e * (w_e - w_i) * sum_square_beam + 2.0 * w_i**2 * sum_square_ejected
    )
    momentum_change = total_momentum[:, -1] - total_momentum[:, 0]
    # Change of the total momentum if the momentum changes of the electrons were not applied
    # (scale of the errors that the check detects)
    unapplied = -w_e * np.array(
        [
            np.sum(ad["electrons", f"particle_momentum_{d}"].v - p_beam_initial[i])
            for i, d in enumerate("xyz")
        ]
    )
    print(f"\nChange of the total momentum at step {step}: {momentum_change} [kg m/s]")
    print(f"Expected: 0 +/- {momentum_uncertainty} [kg m/s] (1 sigma)")
    print(f"(change if the momentum of the electrons was not updated: {unapplied})")
    assert np.all(
        np.abs(momentum_change)
        <= 5.0 * momentum_uncertainty + 1.0e-12 * np.linalg.norm(total_momentum[:, 0])
    )
