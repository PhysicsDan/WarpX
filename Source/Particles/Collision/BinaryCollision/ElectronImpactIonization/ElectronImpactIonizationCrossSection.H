/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRON_IMPACT_IONIZATION_CROSS_SECTION_H_
#define WARPX_ELECTRON_IMPACT_IONIZATION_CROSS_SECTION_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

/**
 * \brief Computes the electron-impact ionization cross section of an ion in a given charge
 * state, using the empirical formula of Lotz (Z. Physik 206, 205 (1967); Z. Physik 216,
 * 241 (1968)), summed over the occupied subshells of the ion:
 *
 * \f[ \sigma = a \sum_s q_s \frac{\ln(E/P_s)}{E P_s} \f]
 *
 * where \f$ a = 4.5 \times 10^{-14} \f$ cm\f$^2\f$ eV\f$^2\f$, \f$ q_s \f$ is the number of
 * electrons in subshell \f$ s \f$, \f$ P_s \f$ its binding energy and \f$ E \f$ the kinetic
 * energy of the incident electron in the rest frame of the ion.
 *
 * Lotz's formula is non-relativistic. Here, it is extended to relativistic incident electrons
 * in the same way as in the relativistic binary-encounter-Bethe model of Kim et al.
 * (Phys. Rev. A 62, 052710 (2000)): \f$ E \f$ is replaced by \f$ m_e c^2 \beta^2 / 2 \f$ in the
 * denominator, and the relativistic correction
 * \f$ \delta = \ln(\beta^2 \gamma^2 m_e c^2 / 2E) - \beta^2 = \ln((\gamma + 1)/2) - \beta^2 \f$,
 * multiplied by the threshold factor \f$ 1 - P_s^2/E^2 \f$, is added to \f$ \ln(E/P_s) \f$.
 * The cross section thus has the asymptotic behavior of the relativistic Bethe formula at high
 * energy, \f$ \ln(\beta^2 \gamma^2 m_e c^2 / 2P_s) - \beta^2 \f$, remains positive above the
 * threshold, and reduces to the original formula of Lotz when \f$ E \ll m_e c^2 \f$.
 *
 * @param[in] kinetic_energy_eV kinetic energy of the incident electron, in the rest frame
 *            of the ion [eV]
 * @param[in] n_subshells number of occupied subshells of the ion
 * @param[in] occupation number of electrons in each occupied subshell
 * @param[in] binding_energy_eV binding energy of each occupied subshell [eV],
 *            sorted in increasing order
 * @return the ionization cross section [m^2]
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal
LotzIonizationCrossSection (amrex::ParticleReal const kinetic_energy_eV,
                            int const n_subshells,
                            amrex::ParticleReal const* AMREX_RESTRICT occupation,
                            amrex::ParticleReal const* AMREX_RESTRICT binding_energy_eV)
{
    // Lotz constant, 4.5e-14 cm^2 eV^2 converted to m^2 eV^2
    constexpr double lotz_constant = 4.5e-18;
    constexpr double m_e_c2_eV = PhysConst::m_e_v<double> * PhysConst::c2_v<double>
                                 / PhysConst::q_e_v<double>;

    auto const E = static_cast<double>(kinetic_energy_eV);
    double const gamma = 1.0 + E/m_e_c2_eV;
    double const beta_sq = E*(E + 2.0*m_e_c2_eV)/((E + m_e_c2_eV)*(E + m_e_c2_eV));
    // m_e c^2 beta^2 / 2, which is equal to E for E << m_e c^2
    double const E_beta = 0.5*E*(gamma + 1.0)/(gamma*gamma);
    // Relativistic correction to the logarithmic term, which is of order E/(m_e c^2)
    // for E << m_e c^2
    double const delta = std::log1p(0.5*E/m_e_c2_eV) - beta_sq;

    double sum = 0.0;
    for (int s = 0; s < n_subshells; ++s) {
        auto const P = static_cast<double>(binding_energy_eV[s]);
        // The binding energies are sorted in increasing order: the remaining
        // subshells cannot be ionized either
        if (E <= P) { break; }
        double const log_term = std::log(E/P) + (1.0 - (P/E)*(P/E))*delta;
        // The logarithmic term is positive above the threshold (since delta > -1/2),
        // but this is enforced here to protect against round-off errors
        if (log_term > 0.0) {
            sum += static_cast<double>(occupation[s])*log_term/P;
        }
    }

    // Below the ionization threshold (including for E = 0, for which E_beta = 0)
    if (sum <= 0.0) { return amrex::ParticleReal(0.); }

    return static_cast<amrex::ParticleReal>(lotz_constant*sum/E_beta);
}

/**
 * \brief Samples the kinetic energy of the electron ejected in an ionization event, in the
 * rest frame of the ion, using the binary-encounter (Thomson) approximation, in which the
 * energy transferred from the incident electron \f$ W = \epsilon + I \f$ (with \f$ \epsilon \f$
 * the kinetic energy of the ejected electron and \f$ I \f$ the ionization energy) is
 * distributed as \f$ dP/dW \propto 1/W^2 \f$. The ejected electron is by convention the
 * slower of the two outgoing electrons, so that \f$ W \f$ is sampled between \f$ I \f$ and
 * \f$ (E + I)/2 \f$, with \f$ E \f$ the kinetic energy of the incident electron.
 *
 * @param[in] kinetic_energy kinetic energy of the incident electron [J]
 * @param[in] ionization_energy ionization energy of the ion [J]
 * @param[in] random_number a random number uniformly distributed in [0, 1)
 * @return the kinetic energy of the ejected electron [J]
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
double
SampleBinaryEncounterEjectedEnergy (double const kinetic_energy, double const ionization_energy,
                                    double const random_number)
{
    if (kinetic_energy <= ionization_energy) { return 0.0; }

    double const inv_W_min = 1.0/ionization_energy;
    double const inv_W_max = 2.0/(kinetic_energy + ionization_energy);
    double const W = 1.0/(inv_W_min - random_number*(inv_W_min - inv_W_max));

    return amrex::max(W - ionization_energy, 0.0);
}

#endif // WARPX_ELECTRON_IMPACT_IONIZATION_CROSS_SECTION_H_
