#!/usr/bin/env python3
#
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This script checks the results of the tests `inputs_test_1d_ion_impact_ionization` and
# `inputs_test_1d_ion_impact_ionization_molecular`, in which a uniform beam of protons ionizes a
# gas of argon atoms (macroparticles not split) or of N2 molecules (macroparticles split, total
# cross section read from a file). It checks that:
# - the populations of the charge states of argon agree with the solution of the ionization rate
#   equations, using the cross sections of the Rudd model (computed independently below, with the
#   binding energies implemented in WarpX), and the fraction of ionized N2 molecules agrees with
#   the cross section of the file,
# - the charge of the ejected electrons balances the charge gained by the targets,
# - the kinetic energy lost by the protons and the targets is equal to the binding energy of the
#   ionized subshells plus the kinetic energy of the ejected electrons (on average),
# - the kinetic energy of the ejected electrons agrees with the Rudd differential cross section,
# - the fast ejected electrons are emitted at the angle of a binary collision with a free
#   electron, in the proportion given by the binary term of the Rudd model,
# - at each output step, the total charge is conserved exactly, the total kinetic energy decreases
#   by the binding energy of the ionized subshells, and the total momentum is conserved: exactly
#   when the macroparticles are split, and within the statistical bounds of the default method
#   otherwise (in which energy and momentum are only conserved on average).

import glob
import os

import numpy as np
import yt
from scipy import constants, integrate
from scipy.linalg import expm

yt.funcs.mylog.setLevel(0)

# Parameters of the simulation, read from the input file
inputs = {}
with open("warpx_used_inputs") as f:
    for line in f:
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            inputs[key.strip()] = value.split("#")[0].strip()
n_p = float(inputs["my_constants.n_p"])  # beam density [m^-3]
E_beam = float(inputs["my_constants.E_beam"])  # beam kinetic energy [eV]
dt = float(inputs["warpx.const_dt"])  # time step [s]
split_macroparticles = inputs.get("ionization.split_macroparticles", "0") in [
    "1",
    "true",
]
molecular = "ionization.binding_energies" in inputs
target = inputs["ionization.species"].split()[1]
product_ion = inputs["ionization.product_species"].split()[0] if molecular else target
print(
    f"Molecular target: {molecular}, macroparticles are split: {split_macroparticles}"
)

# Kinematics of the protons
m_e_eV = constants.m_e * constants.c**2 / constants.e
m_p_eV = constants.m_p * constants.c**2 / constants.e
rydberg = constants.physical_constants["Rydberg constant times hc in eV"][0]
r_e = constants.physical_constants["classical electron radius"][0]
gamma = 1.0 + E_beam / m_p_eV
beta_sq = 1.0 - 1.0 / gamma**2
v_beam = constants.c * np.sqrt(beta_sq)
T = (
    0.5 * m_e_eV * beta_sq
)  # kinetic energy of an electron with the velocity of the protons
mass_ratio = constants.m_e / constants.m_p
W_free = 2 * m_e_eV * beta_sq * gamma**2 / (1 + 2 * gamma * mass_ratio + mass_ratio**2)
delta = np.log(gamma**2) - beta_sq  # relativistic correction of the dipole term

# Parameters of the Rudd model (A1, B1, C1, D1, E1, A2, B2, C2, D2, alpha), from Table V of
# Rudd et al., Rev. Mod. Phys. 64, 441 (1992), and parameters used in WarpX for one-electron targets
parameters_He = [1.02, 2.4, 0.70, 1.15, 0.70, 0.84, 6.0, 0.70, 0.50, 0.86]
parameters_Ne = [0.58, 65.0, 0.23, 0.55, 0.16, 1.40, 0.0, 0.72, 1.35, 0.57]
parameters_Ar = [1.20, 8.0, 0.86, 0.0, 0.80, 0.90, 2.7, 0.75, 0.80, 0.71]
parameters_inner = [1.25, 0.50, 1.00, 1.00, 3.0, 1.10, 1.30, 1.00, 0.0, 0.66]
parameters_H = [0.75, 2.4, 0.70, 1.15, 0.70, 1.2, 6.0, 0.70, 0.50, 0.86]

# Ejected electrons faster than this energy are used to check the emission angles
W_cut = 300.0  # [eV]


def rudd_subshell(B, N, p):
    """Cross section [m^2] of a subshell with N electrons of binding energy B [eV] in the Rudd
    model, for the protons of the beam, and the following quantities for the ejected electrons:
    mean and mean square kinetic energy W, probability that W > W_cut, and probability that
    W > W_cut and that W is sampled from the binary term F2 w/(1+w)^3 of the model."""
    A1, B1, C1, D1, E1, A2, B2, C2, D2, alpha = p
    v_sq = T / B
    v = np.sqrt(v_sq)
    F1 = C1 * v**D1 / (1 + E1 * v ** (D1 + 4)) + A1 * (np.log1p(v_sq) + delta) / (
        v_sq + B1 / v_sq
    )
    L2 = C2 * v**D2
    H2 = A2 / v_sq + B2 / v_sq**2
    F2 = L2 * H2 / (L2 + H2)
    w_c = W_free / B - 2 * v - rydberg / (4 * B)
    w_up = min((E_beam - B) / B, max(w_c, 0) + 40 * v / alpha)

    def integral(f, lower=0.0):
        """Integral from lower to w_up of f(w) / (1+w)^3 times the cutoff factor of the model"""
        if lower >= w_up:
            return 0.0
        return integrate.quad(
            lambda w: (
                f(w) / (1 + w) ** 3 / (1 + np.exp(min(alpha * (w - w_c) / v, 700)))
            ),
            lower,
            w_up,
            points=[w_c] if lower < w_c < w_up else None,
            limit=500,
        )[0]

    total = integral(lambda w: F1 + F2 * w)
    S = np.pi * r_e**2 * N * (m_e_eV / B) ** 2
    mean_W = B * integral(lambda w: (F1 + F2 * w) * w) / total
    mean_W2 = B**2 * integral(lambda w: (F1 + F2 * w) * w**2) / total
    above = integral(lambda w: F1 + F2 * w, W_cut / B)
    above_binary = integral(lambda w: F2 * w, W_cut / B)
    return S * total, mean_W, mean_W2, above / total, above_binary / total


# Subshells (occupation, binding energy) and parameters of the Rudd model of each charge state
if molecular:
    binding = [float(x) for x in inputs["ionization.binding_energies"].split()]
    occupation = [float(x) for x in inputs["ionization.occupations"].split()]
    outer = [float(x) for x in inputs["ionization.rudd_parameters"].split()]
    subshells_of_charge_state = [list(zip(occupation, binding))]
    parameters_of_charge_state = [outer]
else:
    # Ionization energies of argon [eV] (see Source/Utils/Physics/IonizationEnergiesTable.H) and
    # binding energies of the electrons of the neutral argon atom, from the most bound
    # (see Source/Utils/Physics/binding_energy_data.txt)
    ionization_energies = [
        15.7596117, 27.62967, 40.735, 59.58, 74.84, 91.29, 124.41, 143.4567, 422.6,
        479.76, 540.4, 619.0, 685.5, 755.13, 855.5, 918.375, 4120.6656, 4426.2228,
    ]  # fmt: skip
    neutral_binding_energies = [3206.2] * 2 + [324.2] * 2 + [247.74] * 6 + [29.24] * 2
    neutral_binding_energies += [15.76] * 6
    subshells_of_charge_state = []
    parameters_of_charge_state = []
    for q in range(18):
        # The ion keeps the n_electrons most bound electrons of the neutral atom, and the binding
        # energies are shifted such that the binding energy of the outermost electron is the
        # ionization energy of the ion (Carlson et al. 1970)
        n_electrons = 18 - q
        kept = neutral_binding_energies[:n_electrons]
        shift = ionization_energies[q] - kept[-1]
        values, counts = np.unique(kept, return_counts=True)
        subshells_of_charge_state.append(
            [(N, B + shift) for N, B in zip(counts, values)]
        )
        # Parameters of the outer subshells, as a function of the number of electrons
        if n_electrons == 1:
            parameters_of_charge_state.append(parameters_H)
        elif n_electrons < 10:
            parameters_of_charge_state.append(parameters_He)
        elif n_electrons == 10:
            parameters_of_charge_state.append(parameters_Ne)
        else:
            parameters_of_charge_state.append(parameters_Ar)

# For each charge state, cross section and moments of the binding energy B of the ionized subshell
# and of the kinetic energy W of the ejected electron
n_charge_states = len(subshells_of_charge_state)
sigma = np.zeros(n_charge_states)
mean_B, mean_W, var_B, var_W, var_BW = (np.zeros(n_charge_states) for _ in range(5))
fraction_above_cut = np.zeros(n_charge_states)
fraction_binary_above_cut = np.zeros(n_charge_states)
for q, (subshells, outer) in enumerate(
    zip(subshells_of_charge_state, parameters_of_charge_state)
):
    B_min = min(B for N, B in subshells)
    results = np.array(
        [
            rudd_subshell(B, N, parameters_inner if B > 2 * B_min else outer)
            for N, B in subshells
        ]
    )
    B_s = np.array([B for N, B in subshells])
    sigma_s, W_s, W2_s, above_s, binary_s = results.T
    sigma[q] = sigma_s.sum()
    probability = sigma_s / sigma[q]
    mean_B[q] = np.sum(probability * B_s)
    mean_W[q] = np.sum(probability * W_s)
    var_B[q] = np.sum(probability * B_s**2) - mean_B[q] ** 2
    var_W[q] = np.sum(probability * W2_s) - mean_W[q] ** 2
    var_BW[q] = (
        np.sum(probability * (B_s**2 + 2.0 * B_s * W_s + W2_s))
        - (mean_B[q] + mean_W[q]) ** 2
    )
    fraction_above_cut[q] = np.sum(probability * above_s)
    fraction_binary_above_cut[q] = np.sum(probability * binary_s)
print(f"Cross sections of the Rudd model [m^2]: {sigma}")
print(f"Mean binding energy of the ionized subshell [eV]: {mean_B}")
print(f"Mean kinetic energy of the ejected electrons [eV]: {mean_W}")

if molecular:
    # The total cross section is read from the file, and is interpolated linearly
    filename = inputs["ionization.cross_section"]
    energies, values = np.loadtxt(filename, unpack=True)
    sigma_file = np.interp(E_beam, energies, values)
    print(f"Cross section of the file: {sigma_file} [m^2]")
    rates = np.array([n_p * sigma_file * v_beam])
else:
    rates = n_p * sigma * v_beam
print(f"Ionization rates [s^-1]: {rates}")

# Rate equations dN/dt = A N, with N[j] the fraction of targets in charge state j (for a
# molecular target, charge state 1 corresponds to the product ions). Since the rates are the same
# for all the ions, the solution is also the probability that an ion is in charge state j. When
# the macroparticles are split, an ion can be ionized at most once per time step, with
# probability 1 - exp(-rate*dt); for a molecular target, this gives the same evolution as the
# rate equations.
A = np.zeros((n_charge_states + 1, n_charge_states + 1))
for j in range(n_charge_states):
    A[j, j] -= rates[j]
    A[j + 1, j] += rates[j]
assert not split_macroparticles or molecular
N_initial = np.zeros(n_charge_states + 1)
N_initial[0] = 1.0

# Compare the populations of the charge states with the theory, and record at each output step
# the number of ionization events from each charge state and the charge gained by the targets [e]
# (the targets are initially neutral)
plotfiles = sorted(glob.glob(os.path.join("diags", "diag1" + "[0-9]" * 6)))
events_at_step = {0: np.zeros(n_charge_states)}
gained_charge_at_step = {0: 0.0}
for plotfile in plotfiles:
    step = int(plotfile[-6:])
    if step == 0:
        continue
    ad = yt.load(plotfile).all_data()
    if molecular:
        weight_target = ad[target, "particle_weight"].v
        weight_ion = ad[product_ion, "particle_weight"].v
        N_simulation = np.array([weight_target.sum(), weight_ion.sum()])
        n_macroparticles = weight_target.size + weight_ion.size
    else:
        ion_level = ad[target, "particle_ionizationLevel"].v.astype(int)
        ion_weight = ad[target, "particle_weight"].v
        N_simulation = np.array(
            [ion_weight[ion_level == j].sum() for j in range(n_charge_states + 1)]
        )
        n_macroparticles = ion_level.size
    N_simulation /= N_simulation.sum()
    N_theory = expm(A * step * dt) @ N_initial

    # Statistical uncertainty on the fraction of targets in each charge state
    # (all the target macroparticles have the same weight in these tests)
    N_uncertainty = np.sqrt(N_theory * (1.0 - N_theory) / n_macroparticles)

    print(f"\nStep {step}:")
    print(f"  simulated charge state fractions:   {N_simulation}")
    print(f"  solution of the rate equations:     {N_theory}")
    assert np.all(np.abs(N_simulation - N_theory) < 5.0 * N_uncertainty + 1.0e-3)
    if molecular:
        events_at_step[step] = np.array([weight_ion.size])
        gained_charge_at_step[step] = weight_ion.sum()
    else:
        events_at_step[step] = np.array(
            [np.sum(ion_level > q) for q in range(n_charge_states)]
        )
        gained_charge_at_step[step] = np.sum(ion_weight * ion_level)

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
assert int(plotfiles[-1][-6:]) == int(number["step"][-1])

# Number of ionization events from each charge state and weight of the targets
if molecular:
    w_i = weight_ion[0]
    assert np.allclose(weight_ion, w_i, rtol=1.0e-12)
    n_events = np.array([weight_ion.size])
    gained_charge = weight_ion.sum()
    target_species = [target, product_ion]
else:
    w_i = ion_weight[0]
    assert np.allclose(ion_weight, w_i, rtol=1.0e-12)
    n_events = np.array([np.sum(ion_level > q) for q in range(n_charge_states)])
    gained_charge = np.sum(ion_weight * ion_level)
    target_species = [target]
print(f"\nNumber of ionization events from each charge state: {n_events}")
# Weight of the protons (all the macroparticles of the beam initially have the same weight)
w_p = number["protons_weight"][0] / number["protons_macroparticles"][0]

# Charge conservation: each ionization event creates one ejected electron
ejected_weight = number["electrons_weight"][-1]
print(f"\nWeight of the ejected electrons: {ejected_weight}")
print(f"Expected from the charge of the targets: {gained_charge}")
assert np.isclose(ejected_weight, gained_charge, rtol=1.0e-12)
if not split_macroparticles:
    # Each ionization of a target macroparticle creates one ejected electron macroparticle
    assert number["electrons_macroparticles"][-1] == np.sum(n_events)

# The total weight of the protons and of the targets is conserved
weight = number["protons_weight"]
assert np.allclose(weight, weight[0], rtol=1.0e-12)
weight = sum(number[species + "_weight"] for species in target_species)
assert np.allclose(weight, weight[0], rtol=1.0e-12)

# Kinetic energy of the ejected electrons: each ionization event creates an electron of weight
# w_i, whose kinetic energy is sampled from the Rudd differential cross section
ejected_energy = energy["electrons"][-1] / constants.e
expected_ejected_energy = w_i * np.sum(n_events * mean_W)
ejected_energy_uncertainty = w_i * np.sqrt(np.sum(n_events * var_W))
print(f"\nKinetic energy of the ejected electrons: {ejected_energy} [eV]")
print(
    f"Expected: {expected_ejected_energy} +/- {ejected_energy_uncertainty} [eV] (1 sigma)"
)
assert abs(ejected_energy - expected_ejected_energy) < 5.0 * ejected_energy_uncertainty

# Kinetic energy lost by the colliding species: in each ionization event, the proton loses the
# binding energy B of the ionized subshell plus the kinetic energy W of the ejected electron.
# When the macroparticles are split, the energy is conserved exactly for each event, of weight
# w_i. Otherwise, each event is applied to the proton macroparticle with the probability w_i/w_p
# (since w_p > w_i in the test), which conserves the energy on average.
colliding_species = ["protons"] + target_species
energy_lost = (
    sum(energy[species][0] - energy[species][-1] for species in colliding_species)
    / constants.e
)
expected_energy_lost = w_i * np.sum(n_events * (mean_B + mean_W))
if split_macroparticles:
    energy_lost_variance = w_i**2 * np.sum(n_events * var_BW)
else:
    mean_square_BW = var_BW + (mean_B + mean_W) ** 2
    energy_lost_variance = np.sum(
        n_events * (w_p * w_i * mean_square_BW - w_i**2 * (mean_B + mean_W) ** 2)
    )
energy_lost_uncertainty = np.sqrt(energy_lost_variance)
print(f"\nKinetic energy lost by the protons and the targets: {energy_lost} [eV]")
print(f"Expected: {expected_energy_lost} +/- {energy_lost_uncertainty} [eV] (1 sigma)")
assert abs(energy_lost - expected_energy_lost) < 5.0 * energy_lost_uncertainty

# Emission angle of the fast ejected electrons: in a binary collision, the electron is emitted
# at the angle theta given by cos^2(theta) = W (E + M + m_e c^2)^2 / (E (E + 2M) (W + 2 m_e c^2))
# with respect to the direction of the proton (z), while the other electrons are emitted
# isotropically (the targets are at rest, up to their small recoil velocity)
ad = yt.load(plotfiles[-1]).all_data()
ux, uy, uz = (
    ad["electrons", f"particle_momentum_{d}"].v / (constants.m_e * constants.c)
    for d in "xyz"
)
u = np.sqrt(ux**2 + uy**2 + uz**2)
W = m_e_eV * (np.sqrt(1.0 + u**2) - 1.0)
fast = W > W_cut
cos_theta = uz[fast] / u[fast]
total = E_beam + m_p_eV + m_e_eV
cos_theta_binary = np.sqrt(
    W[fast] * total**2 / (E_beam * (E_beam + 2 * m_p_eV) * (W[fast] + 2 * m_e_eV))
)
binary_fraction = np.mean(np.abs(cos_theta - cos_theta_binary) < 1.0e-3)
expected_binary_fraction = np.sum(n_events * fraction_binary_above_cut) / np.sum(
    n_events * fraction_above_cut
)
binary_fraction_uncertainty = np.sqrt(
    expected_binary_fraction * (1.0 - expected_binary_fraction) / fast.sum()
)
print(
    f"\nFraction of the {fast.sum()} electrons faster than {W_cut} eV emitted at the "
    f"binary-collision angle: {binary_fraction}"
)
print(
    f"Expected: {expected_binary_fraction} +/- {binary_fraction_uncertainty} (1 sigma)"
)
assert fast.sum() > 50
assert (
    abs(binary_fraction - expected_binary_fraction)
    < 5.0 * binary_fraction_uncertainty + 0.01
)

# Conservation of the total charge, kinetic energy and momentum, at each output step:
# - charge: the charge gained by the targets is that of the ejected electrons (exactly);
# - energy: the total kinetic energy decreases by the binding energy B of the ionized subshell in
#   each event, of weight w_i. When the macroparticles are split, this holds exactly for each
#   event, but B is only known on average for each charge state (mean_B, var_B), which sets the
#   uncertainty. With the default method, the energy change -(B + W) of the proton is applied to
#   its macroparticle, of weight w_p, with the probability w_i/w_p, and the kinetic energy W' of
#   the ejected electron is sampled independently of W, when the electron is created after the
#   collisions: the energy change of an event, w_i W' - w_p X (B + W) (X = 1 with the probability
#   w_i/w_p, 0 otherwise), is -w_i B on average, with the variance
#   w_i^2 var(W) + w_p w_i <(B + W)^2> - w_i^2 <B + W>^2;
# - momentum: when the macroparticles are split, it is conserved exactly in the rest frame of the
#   target, which is here the lab frame since the molecules are at rest. With the default method,
#   the total momentum changes in each event by w_i (p' - p) + (w_p X - w_i) d, where d is the
#   momentum change of the proton, p the momentum of the ejected electron used to compute d and
#   the recoil of the target, and p' that of the ejected electron created (sampled independently):
#   zero on average, with the variance 2 w_i^2 var(p) + (w_p w_i - w_i^2) d^2 for each component
#   (checked at the last step).
total_momentum = np.array(
    [momentum["total_x"], momentum["total_y"], momentum["total_z"]]
)
proton_charge = number["protons_weight"][0]  # [e]
mean_square_BW = var_BW + (mean_B + mean_W) ** 2
print("\nConservation laws at each output step:")
for k, step in enumerate(number["step"].astype(int)):
    events = events_at_step[step]
    charge_change = (
        number["protons_weight"][k]
        + gained_charge_at_step[step]
        - number["electrons_weight"][k]
        - proton_charge
    )
    energy_change = (energy["total"][k] - energy["total"][0]) / constants.e
    expected_energy_change = -w_i * np.sum(events * mean_B)
    if split_macroparticles:
        energy_variance = w_i**2 * np.sum(events * var_B)
    else:
        energy_variance = np.sum(
            events
            * (
                w_i**2 * var_W
                + w_p * w_i * mean_square_BW
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
    assert abs(charge_change) < 1.0e-12 * proton_charge
    assert abs(energy_change - expected_energy_change) <= 5.0 * energy_uncertainty + (
        1.0e-12 * energy["total"][0] / constants.e
    )
    if split_macroparticles:
        assert relative_momentum_change < 1.0e-10

if not split_macroparticles:
    # The sum of d^2 over the events is estimated from the momentum changes of the protons at the
    # last step, whose sum of squares is (w_i/w_p) sum(d^2) (each event being applied with the
    # probability w_i/w_p), and the sum of var(p) from the momenta of the ejected electrons
    step = int(number["step"][-1])
    ad = yt.load(os.path.join("diags", f"diag2{step:06d}")).all_data()
    p_proton_initial = np.array(
        [0.0, 0.0, constants.m_p * constants.c * np.sqrt(gamma**2 - 1.0)]
    )
    proton_momentum_change = [
        ad["protons", f"particle_momentum_{d}"].v - p_proton_initial[i]
        for i, d in enumerate("xyz")
    ]
    sum_square_protons = np.array([np.sum(dp**2) for dp in proton_momentum_change])
    sum_square_ejected = np.array(
        [
            np.sum(
                (
                    ad["electrons", f"particle_momentum_{d}"].v
                    - np.mean(ad["electrons", f"particle_momentum_{d}"].v)
                )
                ** 2
            )
            for d in "xyz"
        ]
    )
    momentum_uncertainty = np.sqrt(
        w_p * (w_p - w_i) * sum_square_protons + 2.0 * w_i**2 * sum_square_ejected
    )
    momentum_change = total_momentum[:, -1] - total_momentum[:, 0]
    # Change of the total momentum if the momentum changes of the protons were not applied
    # (scale of the errors that the check detects)
    unapplied = -w_p * np.array([np.sum(dp) for dp in proton_momentum_change])
    print(f"\nChange of the total momentum at step {step}: {momentum_change} [kg m/s]")
    print(f"Expected: 0 +/- {momentum_uncertainty} [kg m/s] (1 sigma)")
    print(f"(change if the momentum of the protons was not updated: {unapplied})")
    assert np.all(
        np.abs(momentum_change)
        <= 5.0 * momentum_uncertainty + 1.0e-12 * np.linalg.norm(total_momentum[:, 0])
    )
