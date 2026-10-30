/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_IMPACT_IONIZATION_INITIALIZE_MOMENTUM_H_
#define WARPX_IMPACT_IONIZATION_INITIALIZE_MOMENTUM_H_

#include "ImpactIonizationCrossSection.H"

#include "Utils/ParticleUtils.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>

/**
 * \brief Computes the momenta of the particles that result from the ionization of a target by a
 * projectile (an electron or an ion): the scattered projectile, the ionized target and the
 * ejected electron.
 *
 * The calculation is done in the rest frame of the target. The ionized subshell is selected
 * with a probability proportional to its contribution to the cross section, and the momentum of
 * the ejected electron is sampled with ImpactIonization::SampleEjectedElectron.
 * The projectile loses the binding energy of the subshell plus the kinetic energy of the ejected
 * electron. In a close (binary) collision, the scattered projectile is deflected such that the
 * momentum of the projectile and of the ejected electron is conserved, while in a distant
 * collision it keeps its direction. The target recoils such that the momentum is conserved
 * exactly, and its (small) recoil energy is taken from the scattered projectile, so that the
 * kinetic energy decreases exactly by the binding energy of the subshell.
 *
 * @param[in] atomic_data atomic data of the target
 * @param[in] charge_state charge state (ionization level) of the target before the event
 * @param[in] u1x_in,u1y_in,u1z_in momentum per unit mass (gamma*v) of the projectile
 * @param[in] u2x_in,u2y_in,u2z_in momentum per unit mass (gamma*v) of the target
 * @param[in] m1,m2 masses of the projectile and of the target
 * @param[in] energy_model model for the kinetic energy of the ejected electron
 * @param[out] u1x_out,u1y_out,u1z_out momentum per unit mass of the scattered projectile
 * @param[out] u2x_out,u2y_out,u2z_out momentum per unit mass of the target
 * @param[out] u3x_out,u3y_out,u3z_out momentum per unit mass of the ejected electron
 * @param[out] binding_energy binding energy of the ionized subshell [eV]
 * @param[in] engine the random engine
 * \return whether the projectile has enough energy to ionize the target. This can only be false
 *         because of round-off errors, for a projectile at the ionization threshold. In that case,
 *         the target keeps its momentum and the projectile and the electron are at rest in the
 *         rest frame of the target, i.e. the kinetic energy of the projectile, which is
 *         approximately the ionization energy, is used to ionize the target.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
bool ImpactIonizationInitializeMomentum (
    ImpactIonization::AtomicData const& atomic_data, int const charge_state,
    amrex::ParticleReal const u1x_in, amrex::ParticleReal const u1y_in,
    amrex::ParticleReal const u1z_in,
    amrex::ParticleReal const u2x_in, amrex::ParticleReal const u2y_in,
    amrex::ParticleReal const u2z_in,
    amrex::ParticleReal const m1, amrex::ParticleReal const m2,
    EjectedElectronEnergyModel const energy_model,
    amrex::ParticleReal& u1x_out, amrex::ParticleReal& u1y_out, amrex::ParticleReal& u1z_out,
    amrex::ParticleReal& u2x_out, amrex::ParticleReal& u2y_out, amrex::ParticleReal& u2z_out,
    amrex::ParticleReal& u3x_out, amrex::ParticleReal& u3y_out, amrex::ParticleReal& u3z_out,
    double& binding_energy,
    amrex::RandomEngine const& engine)
{
    using ImpactIonization::m_e_c2_eV;
    constexpr auto c = PhysConst::c_v<double>;
    constexpr auto m_e = PhysConst::m_e_v<double>;

    // Rest energies of the projectile and of the target [eV], and ratios of the electron mass
    // to these masses, used to convert the momenta per unit mass to momenta in units of m_e c
    double const ratio_1 = m_e/static_cast<double>(m1);
    double const ratio_2 = m_e/static_cast<double>(m2);
    double const M1 = m_e_c2_eV/ratio_1;
    double const M2 = m_e_c2_eV/ratio_2;

    // Transform the projectile to the rest frame of the target, and normalize its momentum by
    // m_e c
    amrex::ParticleReal ux = u1x_in;
    amrex::ParticleReal uy = u1y_in;
    amrex::ParticleReal uz = u1z_in;
    ParticleUtils::doLorentzTransformWithU(ux, uy, uz, u2x_in, u2y_in, u2z_in);
    double const pin_x = static_cast<double>(ux)/(c*ratio_1);
    double const pin_y = static_cast<double>(uy)/(c*ratio_1);
    double const pin_z = static_cast<double>(uz)/(c*ratio_1);
    double const pin_sq = pin_x*pin_x + pin_y*pin_y + pin_z*pin_z;
    double const pin = std::sqrt(pin_sq);
    // Kinetic energy of the projectile [eV], written to avoid cancellations
    double const u_sq = pin_sq*ratio_1*ratio_1;
    double const E_in = M1*u_sq/(std::sqrt(1.0 + u_sq) + 1.0);

    // Default outputs, for the case where no subshell can be ionized: the target keeps its
    // momentum and the projectile and the electron are at rest in the rest frame of the target
    u2x_out = u2x_in;
    u2y_out = u2y_in;
    u2z_out = u2z_in;
    u3x_out = 0.;
    u3y_out = 0.;
    u3z_out = 0.;
    ParticleUtils::doLorentzTransformWithU(u3x_out, u3y_out, u3z_out, -u2x_in, -u2y_in, -u2z_in);
    u1x_out = u3x_out;
    u1y_out = u3y_out;
    u1z_out = u3z_out;
    binding_energy = 0.0;
    if (pin <= 0.0) { return false; }

    double const nx = pin_x/pin;
    double const ny = pin_y/pin;
    double const nz = pin_z/pin;

    // Ionized subshell and momentum of the ejected electron (in units of m_e c)
    double B, W, pej_x, pej_y, pej_z;
    bool is_binary;
    ImpactIonization::SampleEjectedElectron(atomic_data, E_in, charge_state,
        nx, ny, nz, energy_model, B, W, pej_x, pej_y, pej_z, is_binary, engine);
    double const E_available = E_in - B - W;
    if (B <= 0.0 || E_available < 0.0) { return false; }
    binding_energy = B;

    // Direction of the scattered projectile
    double nsx = nx;
    double nsy = ny;
    double nsz = nz;
    if (is_binary) {
        double const dx = pin_x - pej_x;
        double const dy = pin_y - pej_y;
        double const dz = pin_z - pej_z;
        double const d = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (d > 0.0) {
            nsx = dx/d;
            nsy = dy/d;
            nsz = dz/d;
        }
    }

    // The target takes the momentum that is not carried by the projectile and the electron, and
    // its recoil energy is taken from the scattered projectile. Since the recoil energy itself
    // depends on the momentum of the scattered projectile, it is computed with fixed-point
    // iterations, which converge quickly since the error is reduced at each iteration by the
    // ratio of the velocity of the target to that of the projectile.
    double E_recoil = 0.0;
    double ps = 0.0;
    double u2x = 0.0, u2y = 0.0, u2z = 0.0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        double const T_scattered = amrex::max(E_available - E_recoil, 0.0);
        // Momentum of the scattered projectile, in units of m_e c
        ps = std::sqrt(T_scattered*(T_scattered + 2.0*M1))/m_e_c2_eV;
        // Momentum per unit mass of the target, in units of c
        u2x = ratio_2*(pin_x - ps*nsx - pej_x);
        u2y = ratio_2*(pin_y - ps*nsy - pej_y);
        u2z = ratio_2*(pin_z - ps*nsz - pej_z);
        double const u2_sq = u2x*u2x + u2y*u2y + u2z*u2z;
        E_recoil = M2*u2_sq/(std::sqrt(1.0 + u2_sq) + 1.0);
    }

    u1x_out = static_cast<amrex::ParticleReal>(ps*ratio_1*nsx*c);
    u1y_out = static_cast<amrex::ParticleReal>(ps*ratio_1*nsy*c);
    u1z_out = static_cast<amrex::ParticleReal>(ps*ratio_1*nsz*c);
    u2x_out = static_cast<amrex::ParticleReal>(u2x*c);
    u2y_out = static_cast<amrex::ParticleReal>(u2y*c);
    u2z_out = static_cast<amrex::ParticleReal>(u2z*c);
    u3x_out = static_cast<amrex::ParticleReal>(pej_x*c);
    u3y_out = static_cast<amrex::ParticleReal>(pej_y*c);
    u3z_out = static_cast<amrex::ParticleReal>(pej_z*c);

    // Transform the three particles back to the lab frame
    ParticleUtils::doLorentzTransformWithU(u1x_out, u1y_out, u1z_out, -u2x_in, -u2y_in, -u2z_in);
    ParticleUtils::doLorentzTransformWithU(u2x_out, u2y_out, u2z_out, -u2x_in, -u2y_in, -u2z_in);
    ParticleUtils::doLorentzTransformWithU(u3x_out, u3y_out, u3z_out, -u2x_in, -u2y_in, -u2z_in);
    return true;
}

/**
 * \brief Computes the momentum of an electron ejected in an ionization event, without modifying
 * the projectile and the target. This is used when the macroparticles are not split, in which
 * case the momenta of the projectile and of the target are updated separately, such that the
 * energy and the momentum are conserved on average.
 *
 * @param[in] atomic_data atomic data of the target
 * @param[in] charge_state charge state (ionization level) of the target before the event
 * @param[in] E kinetic energy of the projectile in the rest frame of the target [eV]
 * @param[in] u1x,u1y,u1z momentum per unit mass (gamma*v) of the projectile, used to define its
 *            direction
 * @param[in] u2x,u2y,u2z momentum per unit mass (gamma*v) of the target
 * @param[in] energy_model model for the kinetic energy of the ejected electron
 * @param[out] u3x_out,u3y_out,u3z_out momentum per unit mass of the ejected electron
 * @param[in] engine the random engine
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ImpactIonizationEjectedMomentum (
    ImpactIonization::AtomicData const& atomic_data, int const charge_state,
    double const E,
    amrex::ParticleReal const u1x, amrex::ParticleReal const u1y, amrex::ParticleReal const u1z,
    amrex::ParticleReal const u2x, amrex::ParticleReal const u2y, amrex::ParticleReal const u2z,
    EjectedElectronEnergyModel const energy_model,
    amrex::ParticleReal& u3x_out, amrex::ParticleReal& u3y_out, amrex::ParticleReal& u3z_out,
    amrex::RandomEngine const& engine)
{
    constexpr auto c = PhysConst::c_v<double>;

    // Direction of the projectile in the rest frame of the target
    amrex::ParticleReal ux = u1x;
    amrex::ParticleReal uy = u1y;
    amrex::ParticleReal uz = u1z;
    ParticleUtils::doLorentzTransformWithU(ux, uy, uz, u2x, u2y, u2z);
    double const u = std::sqrt(static_cast<double>(ux*ux + uy*uy + uz*uz));
    double nx = 0.0, ny = 0.0, nz = 1.0;
    if (u > 0.0) {
        nx = static_cast<double>(ux)/u;
        ny = static_cast<double>(uy)/u;
        nz = static_cast<double>(uz)/u;
    }

    double B, W, pej_x, pej_y, pej_z;
    bool is_binary;
    ImpactIonization::SampleEjectedElectron(atomic_data, E, charge_state,
        nx, ny, nz, energy_model, B, W, pej_x, pej_y, pej_z, is_binary, engine);

    u3x_out = static_cast<amrex::ParticleReal>(pej_x*c);
    u3y_out = static_cast<amrex::ParticleReal>(pej_y*c);
    u3z_out = static_cast<amrex::ParticleReal>(pej_z*c);
    ParticleUtils::doLorentzTransformWithU(u3x_out, u3y_out, u3z_out, -u2x, -u2y, -u2z);
}

#endif // WARPX_IMPACT_IONIZATION_INITIALIZE_MOMENTUM_H_
