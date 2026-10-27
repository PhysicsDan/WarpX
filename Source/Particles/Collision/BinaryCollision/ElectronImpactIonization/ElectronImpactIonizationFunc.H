/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRON_IMPACT_IONIZATION_FUNC_H_
#define WARPX_ELECTRON_IMPACT_IONIZATION_FUNC_H_

#include "ElectronImpactIonizationAtomicData.H"
#include "ElectronImpactIonizationCrossSection.H"
#include "ElectronImpactIonizationInitializeMomentum.H"

#include "Particles/Collision/CollisionFuncBase.H"
#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_DenseBins.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <string>

/**
 * \brief This functor performs the electron-impact ionization of ions within a single cell.
 *  Particles of the incident electron species (first species) and of the target ion species
 *  (second species) are paired with each other and, for each pair, a stochastic process
 *  determines whether the ion is ionized by the electron.
 *
 *  The cross sections are computed for each occupied subshell of the ion (see
 *  ElectronImpactIonizationAtomicData). The rate of ionization events of a pair follows the
 *  pairwise algorithm of Higginson et al., Journal of Computational Physics 388, 439-453 (2019),
 *  which is also used for nuclear fusion and DSMC collisions, and which supports macroparticles
 *  with arbitrary weights: for a pair of macroparticles of weights w1 and w2, the expected
 *  number of events of weight min(w1, w2) during the time step is
 *  x = N_min * max(w1, w2) * sigma * v_rel * dt / dV, where N_min is the number of
 *  macroparticles of the species that has the fewest macroparticles in the cell (which takes
 *  into account the pairs that are not sampled), with relativistic corrections.
 *
 *  Two methods are available:
 *  - If the macroparticles are split (<collision_name>.split_macroparticles = 1), an
 *    ionization event of weight min(w1, w2) occurs with the probability 1 - exp(-x). This weight
 *    is subtracted from both colliding macroparticles, and three new macroparticles are created
 *    in ElectronImpactIonizationCreationFunc (scattered electron, ion and ejected electron),
 *    which conserves the energy and the momentum exactly. Each pair undergoes at most one
 *    ionization event per time step, which requires x << 1 for accuracy.
 *  - Otherwise (default), the macroparticles keep their weight: the events of a pair are
 *    sampled as a Poisson process of rate x/dt, which allows several successive ionizations
 *    within one time step, and each event is applied to the electron with the probability
 *    min(w1, w2)/w1 and to the ion with the probability min(w1, w2)/w2. When an event is applied
 *    to the electron, its momentum is updated (it loses the binding energy of the ionized
 *    subshell and the kinetic energy of the ejected electron); when it is applied to the ion, its
 *    ionization level is increased by one, its recoil momentum is applied, and an electron of
 *    the weight of the ion is created in ElectronImpactIonizationCreationFunc. The energy and the
 *    momentum are conserved on average, and the number of macroparticles only increases by one
 *    per ionization of an ion macroparticle.
 */
class ElectronImpactIonizationFunc
    : public CollisionFuncBase
{
    // Define shortcuts for frequently-used type names
    using ParticleType = WarpXParticleContainer::ParticleType;
    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;
    using SoaData_type = WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

public:
    //! When the macroparticles are not split, p_mask contains the number of ionizations of the
    //! ion of the pair in its lowest bits, and the initial ionization level of the ion above them
    static constexpr int mask_level_shift = 16;
    static constexpr index_type mask_count_bits = (index_type(1) << mask_level_shift) - 1;

    /**
     * \brief Default constructor of the ElectronImpactIonizationFunc class.
     */
    ElectronImpactIonizationFunc () = default;

    /**
     * \brief Constructor of the ElectronImpactIonizationFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     * @param[in] isSameSpecies whether the two colliding species are the same
     *            (must be false)
     */
    ElectronImpactIonizationFunc (std::string const& collision_name,
                                  MultiParticleContainer const * mypc,
                                  bool isSameSpecies);

    struct Executor {
        /**
         * \brief Executor of the ElectronImpactIonizationFunc class. Determines, at the cell
         * level, which pairs of electrons and ions undergo ionization events.
         * When the macroparticles are split, this function does not create the product
         * particles, but instead fills p_mask and the weight of the products, which are used
         * later on when creating the products. When the macroparticles are not split, the
         * momenta of the colliding particles and the ionization level of the ions are updated
         * here, and p_mask and p_product_data are used to create the ejected electrons.
         *
         * @param[in] I1s,I2s is the start index for I1,I2 (inclusive).
         * @param[in] I1e,I2e is the stop index for I1,I2 (exclusive).
         * @param[in] I1,I2 index arrays. They determine all elements that will be used.
         * @param[in,out] soa_1,soa_2 contain the struct of array data of the electrons and of
         * the ions
         * @param[in] m1,m2 are the masses of the electrons and of the ions.
         * @param[in] dt is the time step length between two collision calls.
         * @param[in] dV is the volume of the corresponding cell.
         * @param[in] coll_idx is the collision index offset.
         * @param[in] cell_start_pair is the start index of the pairs in that cell.
         * @param[out] p_mask when the macroparticles are split, 1 for the pairs that undergo
         * an ionization event and 0 otherwise; when they are not split, the number of
         * ionizations of the ion of the pair and its initial ionization level (see
         * mask_level_shift)
         * @param[out] p_pair_indices_1,p_pair_indices_2 arrays that store the indices of the
         * particles of a given pair.
         * @param[out] p_pair_reaction_weight weight of the product particles
         * @param[out] p_product_data when the macroparticles are not split, kinetic energy of
         * the electron in the rest frame of the ion, at the first ionization of the pair [eV]
         * @param[in] engine the random engine.
         */
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        void operator() (
            index_type const I1s, index_type const I1e,
            index_type const I2s, index_type const I2e,
            index_type const* AMREX_RESTRICT I1,
            index_type const* AMREX_RESTRICT I2,
            SoaData_type const& soa_1, SoaData_type const& soa_2,
            GetParticlePosition<PIdx> /*get_position_1*/,
            GetParticlePosition<PIdx> /*get_position_2*/,
            amrex::ParticleReal const /*n1*/, amrex::ParticleReal const /*n2*/,
            amrex::ParticleReal const /*T1*/, amrex::ParticleReal const /*T2*/,
            amrex::Real const /*global_lamdb*/,
            amrex::ParticleReal const /*q1*/, amrex::ParticleReal const /*q2*/,
            amrex::ParticleReal const m1, amrex::ParticleReal const m2,
            amrex::Real const dt, amrex::Real const dV, index_type coll_idx,
            index_type const cell_start_pair, amrex::IntVect const /*global_index*/,
            index_type* AMREX_RESTRICT p_mask,
            index_type* AMREX_RESTRICT p_pair_indices_1,
            index_type* AMREX_RESTRICT p_pair_indices_2,
            amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
            amrex::ParticleReal* AMREX_RESTRICT p_product_data,
            amrex::RandomEngine const& engine) const
        {
            amrex::ParticleReal * const AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];
            uint64_t* AMREX_RESTRICT idcpu1 = soa_1.m_idcpu;

            amrex::ParticleReal * const AMREX_RESTRICT w2 = soa_2.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u2x = soa_2.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u2y = soa_2.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u2z = soa_2.m_rdata[PIdx::uz];
            uint64_t* AMREX_RESTRICT idcpu2 = soa_2.m_idcpu;
            int* AMREX_RESTRICT ion_level = soa_2.m_runtime_idata[m_ionization_level_comp];

            // Number of macroparticles of each species
            index_type const NI1 = I1e - I1s;
            index_type const NI2 = I2e - I2s;
            index_type const max_N = amrex::max(NI1,NI2);
            index_type const min_N = amrex::min(NI1,NI2);

            index_type pair_index = cell_start_pair + coll_idx;

            // Factor of the expected number of events of a pair, which includes the multiplier
            // ratio min_N that takes into account the unsampled pairs
            double const rate_factor = static_cast<double>(min_N)*static_cast<double>(dt)
                                       /static_cast<double>(dV);

            index_type i1 = I1s + coll_idx;
            index_type i2 = I2s + coll_idx;
            // we will start from collision number = coll_idx and then add
            // stride (smaller set size) until we do all collisions (larger set size).
            // Note that the pairs of a given call share the same particle of the species with
            // the fewest particles, and are processed sequentially, which allows the in-place
            // updates of the particles below.
            for (index_type k = coll_idx; k < max_N; k += min_N)
            {
                index_type const j1 = I1[i1];
                index_type const j2 = I2[i2];
                p_mask[pair_index] = index_type(0);

                // do not check for collision if a particle's weight was
                // reduced to zero from a previous collision
                if (idcpu1[j1] != amrex::ParticleIdCpus::Invalid &&
                    idcpu2[j2] != amrex::ParticleIdCpus::Invalid &&
                    ion_level[j2] < m_atomic_data.m_atomic_number)
                {
                    amrex::ParticleReal const w_min = amrex::min(w1[j1], w2[j2]);
                    amrex::ParticleReal const w_max = amrex::max(w1[j1], w2[j2]);

                    if (m_split_macroparticles) {
                        double E_kin;
                        double const x = ExpectedNumberOfEvents(
                            u1x[j1], u1y[j1], u1z[j1], u2x[j2], u2y[j2], u2z[j2],
                            m1, m2, w_max, ion_level[j2], rate_factor, E_kin);
                        // In principle the probability is 1 - exp(-x), but std::expm1 is used
                        // since it maintains correctness for a small x
                        if (x > 0.0 && amrex::Random(engine) < -std::expm1(-x)) {
                            p_mask[pair_index] = index_type(1);
                            p_pair_reaction_weight[pair_index] = w_min;
                            // Remove the reaction weight from the colliding particles' weights
                            BinaryCollisionUtils::remove_weight_from_colliding_particle(
                                w1[j1], idcpu1[j1], w_min);
                            BinaryCollisionUtils::remove_weight_from_colliding_particle(
                                w2[j2], idcpu2[j2], w_min);
                        }
                    } else {
                        int const initial_level = ion_level[j2];
                        double E_first = 0.0;
                        int const n_ionizations = SuccessiveIonizationEvents(
                            u1x[j1], u1y[j1], u1z[j1], u2x[j2], u2y[j2], u2z[j2],
                            m1, m2, w1[j1], w2[j2], ion_level[j2], rate_factor, E_first,
                            engine);
                        if (n_ionizations > 0) {
                            p_mask[pair_index] = static_cast<index_type>(n_ionizations)
                                | (static_cast<index_type>(initial_level) << mask_level_shift);
                            p_pair_reaction_weight[pair_index] = w2[j2];
                            p_product_data[pair_index] = static_cast<amrex::ParticleReal>(E_first);
                        }
                    }
                }

                p_pair_indices_1[pair_index] = j1;
                p_pair_indices_2[pair_index] = j2;
                if (max_N == NI1) { i1 += min_N; }
                if (max_N == NI2) { i2 += min_N; }
                pair_index += min_N;
            }
        }

        /**
         * \brief Computes the kinetic energy of the electron in the rest frame of the ion, and
         * the expected number of ionization events of weight min(w1, w2) for a pair of
         * macroparticles during the time step, following Higginson et al., JCP 388, 439 (2019).
         *
         * @param[in] u1x,u1y,u1z momentum per unit mass (gamma*v) of the electron
         * @param[in] u2x,u2y,u2z momentum per unit mass (gamma*v) of the ion
         * @param[in] m1,m2 masses of the electron and of the ion
         * @param[in] w_max largest weight of the two macroparticles
         * @param[in] charge_state ionization level of the ion
         * @param[in] rate_factor product of the multiplier ratio and dt/dV
         * @param[out] E_kin kinetic energy of the electron in the rest frame of the ion [eV]
         * @return the expected number of events
         */
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        double ExpectedNumberOfEvents (
            amrex::ParticleReal const u1x, amrex::ParticleReal const u1y,
            amrex::ParticleReal const u1z, amrex::ParticleReal const u2x,
            amrex::ParticleReal const u2y, amrex::ParticleReal const u2z,
            amrex::ParticleReal const m1, amrex::ParticleReal const m2,
            amrex::ParticleReal const w_max, int const charge_state,
            double const rate_factor, double& E_kin) const
        {
            amrex::ParticleReal E_kin_COM, v_rel_COM, lab_to_COM_factor;
            BinaryCollisionUtils::get_collision_parameters(
                m1*u1x, m1*u1y, m1*u1z, m2*u2x, m2*u2y, m2*u2z, m1, m2,
                E_kin_COM, v_rel_COM, lab_to_COM_factor);

            // The cross section is a function of the kinetic energy of the electron in the
            // rest frame of the ion, which is obtained from the kinetic energy in the
            // center-of-momentum frame using the Lorentz invariance of
            // s = (E1 + E2)^2 - c^2 (p1 + p2)^2, which gives
            // E_kin_COM*(E_kin_COM + 2*(m1 + m2)*c^2) = 2*m2*c^2*E_kin_ion_frame
            constexpr auto c2 = PhysConst::c2_v<double>;
            auto const E_COM = static_cast<double>(E_kin_COM);
            auto const m1_d = static_cast<double>(m1);
            auto const m2_d = static_cast<double>(m2);
            E_kin = E_COM*(E_COM + 2.0*(m1_d + m2_d)*c2)/(2.0*m2_d*c2)/PhysConst::q_e_v<double>;

            double const sigma = m_atomic_data.CrossSection(E_kin, charge_state);
            return rate_factor*static_cast<double>(lab_to_COM_factor)*static_cast<double>(w_max)
                   *sigma*static_cast<double>(v_rel_COM);
        }

        /**
         * \brief Samples the ionization events of a pair of macroparticles during the time
         * step, when the macroparticles are not split. The events are sampled as a Poisson
         * process whose rate is updated after each event, and each event is applied to the
         * electron and to the ion with the probabilities min(w1, w2)/w1 and min(w1, w2)/w2
         * respectively (using the same random number, so that the event is always applied to
         * at least one of the two particles). The momenta and the ionization level are
         * updated in place.
         *
         * @param[in,out] u1x,u1y,u1z momentum per unit mass (gamma*v) of the electron
         * @param[in,out] u2x,u2y,u2z momentum per unit mass (gamma*v) of the ion
         * @param[in] m1,m2 masses of the electron and of the ion
         * @param[in] w1,w2 weights of the electron and of the ion
         * @param[in,out] charge_state ionization level of the ion
         * @param[in] rate_factor product of the multiplier ratio and dt/dV
         * @param[out] E_first kinetic energy of the electron in the rest frame of the ion when
         *             the ion is ionized for the first time [eV]
         * @param[in] engine the random engine
         * @return the number of ionizations of the ion
         */
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        int SuccessiveIonizationEvents (
            amrex::ParticleReal& u1x, amrex::ParticleReal& u1y, amrex::ParticleReal& u1z,
            amrex::ParticleReal& u2x, amrex::ParticleReal& u2y, amrex::ParticleReal& u2z,
            amrex::ParticleReal const m1, amrex::ParticleReal const m2,
            amrex::ParticleReal const w1, amrex::ParticleReal const w2,
            int& charge_state, double const rate_factor, double& E_first,
            amrex::RandomEngine const& engine) const
        {
            amrex::ParticleReal const w_min = amrex::min(w1, w2);
            amrex::ParticleReal const w_max = amrex::max(w1, w2);
            auto const probability_electron = static_cast<double>(w_min/w1);
            auto const probability_ion = static_cast<double>(w_min/w2);

            int n_ionizations = 0;
            // Remaining time until the end of the time step, in units of dt
            double remaining_time = 1.0;
            // The number of events is bounded since each event either ionizes the ion or
            // decreases the energy of the electron by at least the ionization energy; the
            // maximum below is only a safeguard
            constexpr int max_events = 10000;
            for (int event = 0; event < max_events; ++event) {
                if (charge_state >= m_atomic_data.m_atomic_number) { break; }
                double E_kin;
                double const x = ExpectedNumberOfEvents(u1x, u1y, u1z, u2x, u2y, u2z,
                    m1, m2, w_max, charge_state, rate_factor, E_kin);
                if (x <= 0.0) { break; }

                // Time until the next event, sampled from an exponential distribution
                remaining_time += std::log(static_cast<double>(amrex::Random(engine)))/x;
                if (remaining_time < 0.0) { break; }

                amrex::ParticleReal u1x_new, u1y_new, u1z_new, u2x_new, u2y_new, u2z_new;
                amrex::ParticleReal u3x_new, u3y_new, u3z_new;
                double binding_energy;
                bool const possible = ElectronImpactIonizationInitializeMomentum(
                    m_atomic_data, charge_state, u1x, u1y, u1z, u2x, u2y, u2z, m1, m2,
                    m_ejected_electron_energy_model,
                    u1x_new, u1y_new, u1z_new, u2x_new, u2y_new, u2z_new,
                    u3x_new, u3y_new, u3z_new, binding_energy, engine);
                if (!possible) { break; }

                auto const random_number = static_cast<double>(amrex::Random(engine));
                if (random_number < probability_electron) {
                    u1x = u1x_new;
                    u1y = u1y_new;
                    u1z = u1z_new;
                }
                if (random_number < probability_ion) {
                    u2x = u2x_new;
                    u2y = u2y_new;
                    u2z = u2z_new;
                    if (n_ionizations == 0) { E_first = E_kin; }
                    ++charge_state;
                    ++n_ionizations;
                }
            }
            return n_ionizations;
        }

        bool m_computeSpeciesDensities = false;
        bool m_computeSpeciesTemperatures = false;
        bool m_need_product_data = true;

        // Index of the ionization level in the runtime integer attributes of the ions
        int m_ionization_level_comp = 0;
        // Whether the colliding macroparticles are split in each ionization event
        bool m_split_macroparticles = false;
        // Model for the kinetic energy of the ejected electrons
        EjectedElectronEnergyModel m_ejected_electron_energy_model =
            EjectedElectronEnergyModel::Default;
        // Atomic data of the element of the ions
        ElectronImpactIonization::AtomicData m_atomic_data;
    };

    /**
     * \brief Returns the executor of this functor.
     * The index of the ionization level attribute of the ions is only known once the
     * ionization module of the ion species has been initialized, which happens after
     * this functor is constructed. It is therefore set here.
     */
    [[nodiscard]] Executor executor (int /*lev*/, amrex::MFIter const& /*mfi*/) const
    {
        Executor exe = m_exe;
        exe.m_ionization_level_comp = m_ion_species->GetIntCompIndex("ionizationLevel")
                                      - WarpXParticleContainer::NArrayInt;
        return exe;
    }

    [[nodiscard]] bool use_global_debye_length () const { return false; }

private:

    // Container of the ion species, used to get the index of the ionization level attribute
    WarpXParticleContainer* m_ion_species = nullptr;

    // Atomic data of the element of the ions, used by the executor
    ElectronImpactIonizationAtomicData m_atomic_data;

    Executor m_exe;
};

#endif // WARPX_ELECTRON_IMPACT_IONIZATION_FUNC_H_
