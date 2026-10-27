/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRON_IMPACT_IONIZATION_CROSS_SECTION_H_
#define WARPX_ELECTRON_IMPACT_IONIZATION_CROSS_SECTION_H_

#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>

/**
 * \brief Cross sections and differential cross sections for the ionization of ions by the
 * impact of electrons. All the energies are in eV and are defined in the rest frame of the ion.
 */
namespace ElectronImpactIonization
{
    //! Rest energy of the electron [eV]
    constexpr double m_e_c2_eV = PhysConst::m_e_v<double> * PhysConst::c2_v<double>
                                 / PhysConst::q_e_v<double>;

    //! Number of subshells for which the MBELL model is defined: 1s, 2s, 2p, 3s and 3p
    constexpr int n_mbell_subshells = 5;

    /**
     * \brief Cross section for the ionization of a subshell, from the empirical formula of
     * Lotz (Z. Physik 206, 205 (1967)):
     * \f[ \sigma = a N \frac{\ln(E/B)}{E B} \f]
     * where \f$ a = 4.5 \times 10^{-14} \f$ cm\f$^2\f$ eV\f$^2\f$, \f$ N \f$ is the number of
     * electrons in the subshell and \f$ B \f$ its binding energy. It is extended to relativistic
     * incident electrons in the same way as in the relativistic binary-encounter-Bethe model of
     * Kim et al. (Phys. Rev. A 62, 052710 (2000)): \f$ E \f$ is replaced by
     * \f$ m_e c^2 \beta^2 / 2 \f$ in the denominator, and the relativistic correction
     * \f$ \delta = \ln((\gamma + 1)/2) - \beta^2 \f$, multiplied by the threshold factor
     * \f$ 1 - B^2/E^2 \f$, is added to \f$ \ln(E/B) \f$.
     *
     * @param[in] E kinetic energy of the incident electron [eV]
     * @param[in] N number of electrons in the subshell
     * @param[in] B binding energy of the subshell [eV]
     * @return the cross section [m^2]
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    double LotzCrossSection (double const E, double const N, double const B)
    {
        if (E <= B) { return 0.0; }

        // Lotz constant, 4.5e-14 cm^2 eV^2 converted to m^2 eV^2
        constexpr double lotz_constant = 4.5e-18;

        double const gamma = 1.0 + E/m_e_c2_eV;
        double const beta_sq = E*(E + 2.0*m_e_c2_eV)/((E + m_e_c2_eV)*(E + m_e_c2_eV));
        // m_e c^2 beta^2 / 2, which is equal to E for E << m_e c^2
        double const E_beta = 0.5*E*(gamma + 1.0)/(gamma*gamma);
        // Relativistic correction to the logarithmic term (of order E/(m_e c^2) for small E)
        double const delta = std::log1p(0.5*E/m_e_c2_eV) - beta_sq;
        // This term is positive above the threshold (since delta > -1/2)
        double const log_term = std::log(E/B) + (1.0 - (B/E)*(B/E))*delta;
        if (log_term <= 0.0) { return 0.0; }

        return lotz_constant*N*log_term/(B*E_beta);
    }

    /**
     * \brief Cross section for the ionization of a subshell, from the relativistic
     * binary-encounter-Bethe (RBEB) model of Kim et al. (Phys. Rev. A 62, 052710 (2000)),
     * Eq. (22), in which the average kinetic energy of the bound electrons is approximated by
     * their binding energy.
     *
     * @param[in] E kinetic energy of the incident electron [eV]
     * @param[in] N number of electrons in the subshell
     * @param[in] B binding energy of the subshell [eV]
     * @return the cross section [m^2]
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    double RBEBCrossSection (double const E, double const N, double const B)
    {
        if (E <= B) { return 0.0; }

        constexpr double r_e = PhysConst::r_e_v<double>;
        double const t = E/B;
        double const tp = E/m_e_c2_eV;
        double const bp = B/m_e_c2_eV;
        double const beta_t_sq = 1.0 - 1.0/((1.0 + tp)*(1.0 + tp));
        double const beta_b_sq = 1.0 - 1.0/((1.0 + bp)*(1.0 + bp));
        // The kinetic energy of the bound electrons is approximated by the binding energy
        double const beta_u_sq = beta_b_sq;

        double const prefactor = 2.0*MathConst::pi_v<double>*r_e*r_e*N
                                 /((beta_t_sq + beta_u_sq + beta_b_sq)*bp);
        double const a3 = std::log(beta_t_sq/(1.0 - beta_t_sq)) - beta_t_sq - std::log(2.0*bp);
        double const c1 = (1.0 + 2.0*tp)/((1.0 + 0.5*tp)*(1.0 + 0.5*tp));
        double const c2 = bp*bp/((1.0 + 0.5*tp)*(1.0 + 0.5*tp));

        double const sigma = prefactor*(0.5*a3*(1.0 - 1.0/(t*t)) + 1.0 - 1.0/t
                                        - c1*std::log(t)/(t + 1.0) + 0.5*c2*(t - 1.0));
        return amrex::max(sigma, 0.0);
    }

    /**
     * \brief Cross section for the ionization of a subshell of an ion of atomic number up to 18,
     * from the modified Bell (MBELL) model of Haque et al. (Phys. Scr. 74, 377 (2006)), with the
     * relativistic factor of Gryzinski and the ionic correction factor, as in the EPOCH code
     * (Morris et al., Phys. Plasmas 29, 123907 (2022)).
     *
     * @param[in] E kinetic energy of the incident electron [eV]
     * @param[in] N number of electrons in the subshell
     * @param[in] B binding energy of the subshell [eV]
     * @param[in] subshell index of the subshell: 0, 1, 2, 3 and 4 for 1s, 2s, 2p, 3s and 3p
     * @param[in] atomic_number atomic number of the ion
     * @param[in] q_eff atomic number minus the number of electrons in this subshell and in the
     *            subshells with a larger binding energy
     * @return the cross section [m^2]
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    double MBELLCrossSection (double const E, double const N, double const B,
                              int const subshell, int const atomic_number, int const q_eff)
    {
        if (E <= B || subshell < 0 || subshell >= n_mbell_subshells) { return 0.0; }

        // Parameters of Table 1 of Haque et al. (2006), for the subshells 1s, 2s, 2p, 3s, 3p,
        // in units of 1e-13 eV^2 cm^2
        constexpr double mbell_a[n_mbell_subshells] = {0.525, 0.530, 0.600, 0.130, 0.388};
        constexpr double mbell_b[n_mbell_subshells][7] = {
            {-0.510,  0.2000, 0.0500, -0.025, -0.100, 0.00, 0.00},
            {-0.410,  0.1500, 0.1500, -0.200, -0.150, 0.00, 0.00},
            {-0.400, -0.7100, 0.6550,  0.425, -0.750, 0.00, 0.00},
            { 0.250, -1.5000, 2.4000,  3.220, -3.667, 0.00, 0.00},
            {-0.200, -0.2356, 0.5355,  3.150, -8.500, 5.05, 0.37}};
        // Exponent lambda of the ionic correction factor, for s and p subshells
        constexpr double mbell_lambda[n_mbell_subshells] = {1.270, 1.270, 0.542, 1.270, 0.542};
        constexpr double mbell_m = 3.0;

        double const u = E/B;
        double const J = m_e_c2_eV/B;

        // Bell cross section, in units of 1e-13 cm^2 eV^2 / eV^2
        double b_term = 0.0;
        double power = 1.0;
        for (int k = 0; k < 7; ++k) {
            power *= 1.0 - 1.0/u;
            b_term += mbell_b[subshell][k]*power;
        }
        double const sigma_bell = (mbell_a[subshell]*std::log(u) + b_term)/(B*E);

        // Relativistic factor of Gryzinski
        double const one_j = 1.0 + J;
        double const one_2j = 1.0 + 2.0*J;
        double const u_j = u + J;
        double const u_2j = u + 2.0*J;
        double const gryzinski = one_2j/u_2j*(u_j/one_j)*(u_j/one_j)
            *std::pow((1.0 + u)*u_2j*one_j*one_j/(J*J*one_2j + u*u_2j*one_j*one_j), 1.5);

        // Ionic correction factor
        double const f_ion = 1.0 + mbell_m*std::pow(
            static_cast<double>(q_eff)/(static_cast<double>(atomic_number)*u),
            mbell_lambda[subshell]);

        // Convert from 1e-13 cm^2 eV^2 / eV^2 to m^2
        double const sigma = N*f_ion*gryzinski*sigma_bell*1.e-17;
        return amrex::max(sigma, 0.0);
    }

    /**
     * \brief Light-weight structure that gives access, from GPU kernels, to the atomic data of
     * the element of the ions: occupied subshells of each charge state, and model used for the
     * cross sections. The data is owned by ElectronImpactIonizationAtomicData.
     */
    struct AtomicData
    {
        //! Atomic number of the element
        int m_atomic_number = 0;
        //! Model used for the cross sections
        IonizationCrossSectionModel m_model = IonizationCrossSectionModel::Lotz;
        //! For each charge state, index of its first subshell in the arrays below (size Z+1)
        int const* m_subshell_offsets = nullptr;
        //! Number of electrons in each subshell
        amrex::ParticleReal const* m_occupation = nullptr;
        //! Binding energy of each subshell [eV], in increasing order for a given charge state
        amrex::ParticleReal const* m_binding_energy = nullptr;
        //! For the MBELL model: index of the subshell (0 to 4 for 1s, 2s, 2p, 3s, 3p)
        int const* m_mbell_subshell = nullptr;
        //! For the MBELL model: atomic number minus the number of electrons in the subshell
        //! and in the more bound subshells
        int const* m_mbell_q_eff = nullptr;

        /**
         * \brief Cross section for the ionization of a given subshell
         *
         * @param[in] E kinetic energy of the incident electron [eV]
         * @param[in] s global index of the subshell
         * @return the cross section [m^2]
         */
        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_INLINE
        double SubshellCrossSection (double const E, int const s) const
        {
            auto const N = static_cast<double>(m_occupation[s]);
            auto const B = static_cast<double>(m_binding_energy[s]);
            switch (m_model) {
                case IonizationCrossSectionModel::RBEB:
                    return RBEBCrossSection(E, N, B);
                case IonizationCrossSectionModel::MBELL:
                    return MBELLCrossSection(E, N, B, m_mbell_subshell[s], m_atomic_number,
                                             m_mbell_q_eff[s]);
                default:
                    return LotzCrossSection(E, N, B);
            }
        }

        /**
         * \brief Total cross section for the ionization of an ion in a given charge state
         *
         * @param[in] E kinetic energy of the incident electron [eV]
         * @param[in] charge_state charge state (ionization level) of the ion
         * @return the cross section [m^2]
         */
        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_INLINE
        double CrossSection (double const E, int const charge_state) const
        {
            if (charge_state < 0 || charge_state >= m_atomic_number) { return 0.0; }
            double sigma = 0.0;
            for (int s = m_subshell_offsets[charge_state];
                 s < m_subshell_offsets[charge_state+1]; ++s) {
                // The binding energies are sorted in increasing order: the remaining
                // subshells cannot be ionized either
                if (E <= static_cast<double>(m_binding_energy[s])) { break; }
                sigma += SubshellCrossSection(E, s);
            }
            return sigma;
        }

        /**
         * \brief Randomly selects the subshell that is ionized, with a probability proportional
         * to its contribution to the total cross section.
         *
         * @param[in] E kinetic energy of the incident electron [eV]
         * @param[in] charge_state charge state (ionization level) of the ion
         * @param[in] random_number a random number uniformly distributed in [0, 1)
         * @return the global index of the subshell, or -1 if no subshell can be ionized
         */
        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_INLINE
        int SampleSubshell (double const E, int const charge_state,
                            double const random_number) const
        {
            double const sigma = CrossSection(E, charge_state);
            if (sigma <= 0.0) { return -1; }
            double const target = random_number*sigma;
            double cumulative = 0.0;
            int last = -1;
            for (int s = m_subshell_offsets[charge_state];
                 s < m_subshell_offsets[charge_state+1]; ++s) {
                if (E <= static_cast<double>(m_binding_energy[s])) { break; }
                double const sigma_s = SubshellCrossSection(E, s);
                if (sigma_s > 0.0) {
                    cumulative += sigma_s;
                    last = s;
                    if (cumulative > target) { return s; }
                }
            }
            return last;
        }
    };

    /**
     * \brief Samples the kinetic energy of the electron ejected in an ionization event, from
     * the singly differential cross section of the RBEB model (Kim et al., Phys. Rev. A 62,
     * 052710 (2000), Eq. (19) with the approximations of the BEB model):
     * \f[ \frac{d\sigma}{dw} \propto \frac{1}{(w+1)^2} + \frac{1}{(t-w)^2}
     *     - \frac{C_1}{t+1}\left(\frac{1}{w+1} + \frac{1}{t-w}\right) + C_2
     *     + A_3\left(\frac{1}{(w+1)^3} + \frac{1}{(t-w)^3}\right) \f]
     * where \f$ w = W/B \f$ is the kinetic energy of the ejected electron in units of the
     * binding energy \f$ B \f$ of the subshell, \f$ t = E/B \f$, and
     * \f$ 0 \leq w \leq (t-1)/2 \f$ (the ejected electron is by convention the slower of the two
     * outgoing electrons). The first three terms and the term \f$ C_2 \f$ describe close
     * (binary) collisions, in which the energy is transferred to a quasi-free electron, while
     * the last term describes distant (dipole) collisions, which mostly produce slow electrons.
     * The distribution is sampled by composition and rejection.
     *
     * @param[in] E kinetic energy of the incident electron [eV]
     * @param[in] B binding energy of the subshell [eV]
     * @param[out] is_binary whether the sampled energy corresponds to a close collision
     * @param[in] engine the random engine
     * @return the kinetic energy of the ejected electron [eV]
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    double SampleEjectedElectronEnergy (double const E, double const B, bool& is_binary,
                                        amrex::RandomEngine const& engine)
    {
        is_binary = true;
        double const t = E/B;
        if (t <= 1.0) { return 0.0; }

        double const tp = E/m_e_c2_eV;
        double const bp = B/m_e_c2_eV;
        double const beta_t_sq = 1.0 - 1.0/((1.0 + tp)*(1.0 + tp));
        double const a3 = std::log(beta_t_sq/(1.0 - beta_t_sq)) - beta_t_sq - std::log(2.0*bp);
        double const c1 = (1.0 + 2.0*tp)/((1.0 + 0.5*tp)*(1.0 + 0.5*tp));
        double const c2 = bp*bp/((1.0 + 0.5*tp)*(1.0 + 0.5*tp));
        double const w_max = 0.5*(t - 1.0);
        double const a3_pos = amrex::max(a3, 0.0);
        double const a3_neg = amrex::max(-a3, 0.0);

        // Integrals over [0, w_max] of the positive terms (binary, dipole and constant terms)
        double const tp1 = t + 1.0;
        double const int_binary_1 = (t - 1.0)/tp1;            // 1/(w+1)^2
        double const int_binary_2 = (t - 1.0)/(t*tp1);        // 1/(t-w)^2
        double const int_dipole_1 = 0.5*(1.0 - 4.0/(tp1*tp1)); // 1/(w+1)^3
        double const int_dipole_2 = 0.5*(4.0/(tp1*tp1) - 1.0/(t*t)); // 1/(t-w)^3
        double const weight_binary = int_binary_1 + int_binary_2;
        double const weight_dipole = a3_pos*(int_dipole_1 + int_dipole_2);
        double const weight_constant = c2*w_max;
        double const weight_total = weight_binary + weight_dipole + weight_constant;

        double w = 0.0;
        constexpr int max_trials = 100;
        for (int trial = 0; trial < max_trials; ++trial) {
            // Sample w from the positive terms
            double const r1 = amrex::Random(engine)*weight_total;
            double const r2 = amrex::Random(engine);
            if (r1 < weight_binary) {
                is_binary = true;
                if (r1 < int_binary_1) {
                    w = 1.0/(1.0 - r2*int_binary_1) - 1.0;
                } else {
                    w = t - 1.0/(1.0/t + r2*int_binary_2);
                }
            } else if (r1 < weight_binary + weight_dipole) {
                is_binary = false;
                if (r1 - weight_binary < a3_pos*int_dipole_1) {
                    w = 1.0/std::sqrt(1.0 - 2.0*r2*int_dipole_1) - 1.0;
                } else {
                    w = t - 1.0/std::sqrt(1.0/(t*t) + 2.0*r2*int_dipole_2);
                }
            } else {
                is_binary = true;
                w = r2*w_max;
            }
            w = amrex::min(amrex::max(w, 0.0), w_max);

            // Accept with the probability (positive terms - negative terms)/(positive terms)
            double const inv_w1 = 1.0/(w + 1.0);
            double const inv_tw = 1.0/(t - w);
            double const binary = inv_w1*inv_w1 + inv_tw*inv_tw;
            double const dipole = inv_w1*inv_w1*inv_w1 + inv_tw*inv_tw*inv_tw;
            double const positive = binary + a3_pos*dipole + c2;
            double const negative = c1/tp1*(inv_w1 + inv_tw) + a3_neg*dipole;
            if (amrex::Random(engine)*positive < positive - negative) { break; }
        }

        return w*B;
    }

    /**
     * \brief Samples the momentum of the electron ejected in an ionization event, in the rest
     * frame of the ion: selects the ionized subshell, samples the kinetic energy of the ejected
     * electron and its direction. For close (binary) collisions, the ejected electron is emitted
     * at the angle given by the kinematics of a collision with a free electron at rest, with
     * respect to the direction of the incident electron; for distant collisions, it is emitted
     * isotropically.
     *
     * @param[in] atomic_data atomic data of the element of the ion
     * @param[in] E kinetic energy of the incident electron [eV]
     * @param[in] charge_state charge state (ionization level) of the ion
     * @param[in] nx,ny,nz direction of the incident electron (unit vector)
     * @param[in] energy_model model for the kinetic energy of the ejected electron
     * @param[out] B binding energy of the ionized subshell [eV] (0 if no subshell can be ionized)
     * @param[out] W kinetic energy of the ejected electron [eV]
     * @param[out] px,py,pz momentum of the ejected electron, in units of m_e c
     * @param[out] is_binary whether the event is a close (binary) collision
     * @param[in] engine the random engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void SampleEjectedElectron (AtomicData const& atomic_data, double const E,
                                int const charge_state,
                                double const nx, double const ny, double const nz,
                                EjectedElectronEnergyModel const energy_model,
                                double& B, double& W, double& px, double& py, double& pz,
                                bool& is_binary, amrex::RandomEngine const& engine)
    {
        B = 0.0;
        W = 0.0;
        px = 0.0;
        py = 0.0;
        pz = 0.0;
        is_binary = false;
        int const s = atomic_data.SampleSubshell(E, charge_state,
                                                 static_cast<double>(amrex::Random(engine)));
        if (s < 0) { return; }
        B = static_cast<double>(atomic_data.m_binding_energy[s]);

        if (energy_model == EjectedElectronEnergyModel::AtRest) { return; }
        W = SampleEjectedElectronEnergy(E, B, is_binary, engine);

        // Momentum of the ejected electron, in units of m_e c
        double const T = W/m_e_c2_eV;
        double const p = std::sqrt(T*(T + 2.0));
        double const phi = 2.0*MathConst::pi_v<double>*static_cast<double>(amrex::Random(engine));
        double cos_theta = 0.0;
        if (is_binary) {
            // Angle with respect to the incident electron, from the kinematics of a binary
            // collision with an electron at rest
            double const Ti = E/m_e_c2_eV;
            cos_theta = std::sqrt(amrex::min(T*(Ti + 2.0)/(Ti*(T + 2.0)), 1.0));
        } else {
            cos_theta = 2.0*static_cast<double>(amrex::Random(engine)) - 1.0;
        }
        double const sin_theta = std::sqrt(amrex::max(1.0 - cos_theta*cos_theta, 0.0));

        // Orthonormal basis (e1, e2) perpendicular to the direction of the incident electron
        double e1x = 0.0, e1y = 0.0, e1z = 0.0;
        if (std::abs(nx) < 0.9) {
            // e1 = n x (1, 0, 0) / |...|
            double const norm = std::sqrt(ny*ny + nz*nz);
            e1y = nz/norm;
            e1z = -ny/norm;
        } else {
            // e1 = n x (0, 1, 0) / |...|
            double const norm = std::sqrt(nx*nx + nz*nz);
            e1x = -nz/norm;
            e1z = nx/norm;
        }
        double const e2x = ny*e1z - nz*e1y;
        double const e2y = nz*e1x - nx*e1z;
        double const e2z = nx*e1y - ny*e1x;
        double const cos_phi = std::cos(phi);
        double const sin_phi = std::sin(phi);
        px = p*(cos_theta*nx + sin_theta*(cos_phi*e1x + sin_phi*e2x));
        py = p*(cos_theta*ny + sin_theta*(cos_phi*e1y + sin_phi*e2y));
        pz = p*(cos_theta*nz + sin_theta*(cos_phi*e1z + sin_phi*e2z));
    }
}

#endif // WARPX_ELECTRON_IMPACT_IONIZATION_CROSS_SECTION_H_
