/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRON_IMPACT_IONIZATION_INITIALIZE_MOMENTUM_H_
#define WARPX_ELECTRON_IMPACT_IONIZATION_INITIALIZE_MOMENTUM_H_

#include "ElectronImpactIonizationCrossSection.H"

#include "Particles/Algorithms/KineticEnergy.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>

/**
 * \brief Computes the momenta of the particles that result from the ionization of an ion
 * by an incident electron: the scattered incident electron, the ion and the ejected electron.
 *
 * The calculation is done in the rest frame of the ion. In this frame, the incident electron
 * loses the ionization energy of the ion plus the kinetic energy given to the ejected electron,
 * which is either zero or sampled from the binary-encounter distribution, depending on
 * ejected_electron_energy_model. The scattered electron keeps the direction of the incident
 * electron, while the ejected electron is emitted isotropically. The ion recoils such that the
 * momentum is conserved, and its (small) recoil energy is subtracted from the energy of the
 * scattered electron, so that the kinetic energy minus the ionization energy is conserved.
 *
 * @param[in] u1x_in,u1y_in,u1z_in momentum per unit mass (gamma*v) of the incident electron
 * @param[in] u2x_in,u2y_in,u2z_in momentum per unit mass (gamma*v) of the ion
 * @param[in] m1,m2 masses of the electrons and of the ion
 * @param[in] ionization_energy ionization energy of the ion [J]
 * @param[in] ejected_electron_energy_model model for the energy of the ejected electron
 * @param[out] u1x_out,u1y_out,u1z_out momentum per unit mass of the scattered electron
 * @param[out] u2x_out,u2y_out,u2z_out momentum per unit mass of the ion
 * @param[out] u3x_out,u3y_out,u3z_out momentum per unit mass of the ejected electron
 * @param[in] engine the random engine
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ElectronImpactIonizationInitializeMomentum (
    amrex::ParticleReal const u1x_in, amrex::ParticleReal const u1y_in,
    amrex::ParticleReal const u1z_in,
    amrex::ParticleReal const u2x_in, amrex::ParticleReal const u2y_in,
    amrex::ParticleReal const u2z_in,
    amrex::ParticleReal const m1, amrex::ParticleReal const m2,
    amrex::ParticleReal const ionization_energy,
    EjectedElectronEnergyModel const ejected_electron_energy_model,
    amrex::ParticleReal& u1x_out, amrex::ParticleReal& u1y_out, amrex::ParticleReal& u1z_out,
    amrex::ParticleReal& u2x_out, amrex::ParticleReal& u2y_out, amrex::ParticleReal& u2z_out,
    amrex::ParticleReal& u3x_out, amrex::ParticleReal& u3y_out, amrex::ParticleReal& u3z_out,
    amrex::RandomEngine const& engine)
{
    constexpr auto c = PhysConst::c_v<double>;
    constexpr auto c2 = PhysConst::c2_v<double>;
    auto const m1_d = static_cast<double>(m1);
    auto const m2_d = static_cast<double>(m2);

    // Transform the incident electron to the rest frame of the ion
    amrex::ParticleReal ux = u1x_in;
    amrex::ParticleReal uy = u1y_in;
    amrex::ParticleReal uz = u1z_in;
    ParticleUtils::doLorentzTransformWithU(ux, uy, uz, u2x_in, u2y_in, u2z_in);
    auto const ux_d = static_cast<double>(ux);
    auto const uy_d = static_cast<double>(uy);
    auto const uz_d = static_cast<double>(uz);
    double const u_sq = ux_d*ux_d + uy_d*uy_d + uz_d*uz_d;
    double const u = std::sqrt(u_sq);

    // Kinetic energy of the incident electron in the rest frame of the ion
    double const E_incident = Algorithms::KineticEnergy<double>(ux, uy, uz, m1);

    // Energy left to the two outgoing electrons after the ionization of the ion.
    // It is non-negative since ionization can only occur above the ionization threshold,
    // but this is enforced here to protect against round-off errors.
    auto const I = static_cast<double>(ionization_energy);
    double const E_available = amrex::max(E_incident - I, 0.0);

    double E_ejected = 0.0;
    if (ejected_electron_energy_model == EjectedElectronEnergyModel::BinaryEncounter) {
        E_ejected = SampleBinaryEncounterEjectedEnergy(
            E_incident, I, static_cast<double>(amrex::Random(engine)));
        E_ejected = amrex::min(E_ejected, 0.5*E_available);
    }

    // The ejected electron is emitted isotropically.
    // (The momentum per unit mass of an electron of kinetic energy E is
    // u = sqrt(E*(E + 2*m*c^2))/(m*c).)
    amrex::ParticleReal nx_ejected, ny_ejected, nz_ejected;
    ParticleUtils::getRandomVector(nx_ejected, ny_ejected, nz_ejected, engine);
    double const u_ejected = std::sqrt(E_ejected*(E_ejected + 2.0*m1_d*c2))/(m1_d*c);
    double const u3x = u_ejected*static_cast<double>(nx_ejected);
    double const u3y = u_ejected*static_cast<double>(ny_ejected);
    double const u3z = u_ejected*static_cast<double>(nz_ejected);

    // The scattered electron keeps the direction of the incident electron.
    // (u > 0 since the incident electron is above the ionization threshold.)
    double const nx = ux_d/u;
    double const ny = uy_d/u;
    double const nz = uz_d/u;

    // The ion, initially at rest in this frame, takes the momentum lost by the electrons.
    // Its recoil energy is taken from the energy of the scattered electron. Since the recoil
    // energy itself depends on the momentum of the scattered electron, it is computed with
    // fixed-point iterations. These converge quickly, as the error on the recoil energy is
    // reduced by a factor of order m1/m2 at each iteration.
    double const mass_ratio = m1_d/m2_d;
    double E_recoil = 0.0;
    double u_scattered = 0.0;
    double u2x = 0.0, u2y = 0.0, u2z = 0.0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        double const E_scattered = amrex::max(E_available - E_ejected - E_recoil, 0.0);
        u_scattered = std::sqrt(E_scattered*(E_scattered + 2.0*m1_d*c2))/(m1_d*c);
        u2x = mass_ratio*(ux_d - u_scattered*nx - u3x);
        u2y = mass_ratio*(uy_d - u_scattered*ny - u3y);
        u2z = mass_ratio*(uz_d - u_scattered*nz - u3z);
        double const u2_sq = u2x*u2x + u2y*u2y + u2z*u2z;
        E_recoil = m2_d*u2_sq/(std::sqrt(1.0 + u2_sq/c2) + 1.0);
    }

    u1x_out = static_cast<amrex::ParticleReal>(u_scattered*nx);
    u1y_out = static_cast<amrex::ParticleReal>(u_scattered*ny);
    u1z_out = static_cast<amrex::ParticleReal>(u_scattered*nz);
    u2x_out = static_cast<amrex::ParticleReal>(u2x);
    u2y_out = static_cast<amrex::ParticleReal>(u2y);
    u2z_out = static_cast<amrex::ParticleReal>(u2z);
    u3x_out = static_cast<amrex::ParticleReal>(u3x);
    u3y_out = static_cast<amrex::ParticleReal>(u3y);
    u3z_out = static_cast<amrex::ParticleReal>(u3z);

    // Transform the three particles back to the lab frame
    ParticleUtils::doLorentzTransformWithU(u1x_out, u1y_out, u1z_out, -u2x_in, -u2y_in, -u2z_in);
    ParticleUtils::doLorentzTransformWithU(u2x_out, u2y_out, u2z_out, -u2x_in, -u2y_in, -u2z_in);
    ParticleUtils::doLorentzTransformWithU(u3x_out, u3y_out, u3z_out, -u2x_in, -u2y_in, -u2z_in);
}

#endif // WARPX_ELECTRON_IMPACT_IONIZATION_INITIALIZE_MOMENTUM_H_
