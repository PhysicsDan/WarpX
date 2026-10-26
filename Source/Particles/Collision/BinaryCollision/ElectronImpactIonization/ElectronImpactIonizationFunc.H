/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRON_IMPACT_IONIZATION_FUNC_H_
#define WARPX_ELECTRON_IMPACT_IONIZATION_FUNC_H_

#include "ElectronImpactIonizationCrossSection.H"

#include "Particles/Collision/CollisionFuncBase.H"
#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_DenseBins.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <string>

/**
 * \brief This functor performs the electron-impact ionization of ions within a single cell.
 *  Particles of the incident electron species (first species) and of the target ion species
 *  (second species) are paired with each other and, for each pair, a stochastic process
 *  determines whether the ion is ionized by the electron. If so, p_mask is set to 1 so that
 *  the products of the ionization event can be created in ElectronImpactIonizationCreationFunc.
 *
 *  The ionization cross section of each charge state is computed with the formula of Lotz
 *  (see LotzIonizationCrossSection), using the ionization energies of the element of the ions
 *  and the ground-state electron configuration of each charge state.
 *  The probability of ionization follows the pairwise algorithm of Higginson et al.,
 *  Journal of Computational Physics 388, 439-453 (2019), which is also used for nuclear fusion
 *  and DSMC collisions, and which supports macroparticles with arbitrary weights.
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
         * level, which pairs of electrons and ions undergo an ionization event.
         * Note that this function does not create the product particles, but instead fills
         * p_mask, the weight of the products and the ionization energy of each pair, which
         * are used later on when actually creating the products.
         * The weight of the products is subtracted from the weight of both colliding particles.
         *
         * @param[in] I1s,I2s is the start index for I1,I2 (inclusive).
         * @param[in] I1e,I2e is the stop index for I1,I2 (exclusive).
         * @param[in] I1,I2 index arrays. They determine all elements that will be used.
         * @param[in] soa_1,soa_2 contain the struct of array data of the electrons and of the ions
         * @param[in] m1,m2 are the masses of the electrons and of the ions.
         * @param[in] dt is the time step length between two collision calls.
         * @param[in] dV is the volume of the corresponding cell.
         * @param[in] coll_idx is the collision index offset.
         * @param[in] cell_start_pair is the start index of the pairs in that cell.
         * @param[out] p_mask is set to 1 for the pairs that undergo an ionization event,
         * and to 0 otherwise.
         * @param[out] p_pair_indices_1,p_pair_indices_2 arrays that store the indices of the
         * particles of a given pair.
         * @param[out] p_pair_reaction_weight stores the weight of the product particles.
         * @param[out] p_product_data stores the ionization energy of the ion of each pair [eV].
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
            int const* AMREX_RESTRICT ion_level = soa_2.m_runtime_idata[m_ionization_level_comp];

            // Number of macroparticles of each species
            index_type const NI1 = I1e - I1s;
            index_type const NI2 = I2e - I2s;
            index_type const max_N = amrex::max(NI1,NI2);
            index_type const min_N = amrex::min(NI1,NI2);

            index_type pair_index = cell_start_pair + coll_idx;

            // multiplier ratio to take into account unsampled pairs
            auto const multiplier_ratio = static_cast<int>(min_N);

            index_type i1 = I1s + coll_idx;
            index_type i2 = I2s + coll_idx;
            // we will start from collision number = coll_idx and then add
            // stride (smaller set size) until we do all collisions (larger set size)
            for (index_type k = coll_idx; k < max_N; k += min_N)
            {
                // do not check for collision if a particle's weight was
                // reduced to zero from a previous collision
                if (idcpu1[ I1[i1] ]==amrex::ParticleIdCpus::Invalid ||
                    idcpu2[ I2[i2] ]==amrex::ParticleIdCpus::Invalid ) {
                    p_mask[pair_index] = index_type(0);
                } else {

                    SingleIonizationEvent(
                        u1x[ I1[i1] ], u1y[ I1[i1] ], u1z[ I1[i1] ],
                        u2x[ I2[i2] ], u2y[ I2[i2] ], u2z[ I2[i2] ],
                        m1, m2, w1[ I1[i1] ], w2[ I2[i2] ], ion_level[ I2[i2] ],
                        dt, dV, static_cast<int>(pair_index), multiplier_ratio,
                        p_mask, p_pair_reaction_weight, p_product_data, engine);

                    // Remove pair reaction weight from the colliding particles' weights
                    if (p_mask[pair_index]) {
                        BinaryCollisionUtils::remove_weight_from_colliding_particle(
                            w1[ I1[i1] ], idcpu1[ I1[i1] ], p_pair_reaction_weight[pair_index]);
                        BinaryCollisionUtils::remove_weight_from_colliding_particle(
                            w2[ I2[i2] ], idcpu2[ I2[i2] ], p_pair_reaction_weight[pair_index]);
                    }
                }

                p_pair_indices_1[pair_index] = I1[i1];
                p_pair_indices_2[pair_index] = I2[i2];
                if (max_N == NI1) { i1 += min_N; }
                if (max_N == NI2) { i2 += min_N; }
                pair_index += min_N;
            }
        }

        /**
         * \brief Determines whether a given electron-ion pair undergoes an ionization event.
         *
         * @param[in] u1x,u1y,u1z momentum per unit mass (gamma*v) of the electron
         * @param[in] u2x,u2y,u2z momentum per unit mass (gamma*v) of the ion
         * @param[in] m1,m2 masses of the electron and of the ion
         * @param[in] w1,w2 weights of the electron and of the ion
         * @param[in] ion_level ionization level of the ion
         * @param[in] dt time step length between two collision calls
         * @param[in] dV volume of the cell
         * @param[in] pair_index index of the pair
         * @param[in] multiplier_ratio factor used to take into account unsampled pairs
         * @param[out] p_mask set to 1 if an ionization event occurs for that pair, 0 otherwise
         * @param[out] p_pair_reaction_weight weight of the product particles
         * @param[out] p_product_data ionization energy of the ion [eV]
         * @param[in] engine the random engine
         */
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        void SingleIonizationEvent (
            amrex::ParticleReal const u1x, amrex::ParticleReal const u1y,
            amrex::ParticleReal const u1z, amrex::ParticleReal const u2x,
            amrex::ParticleReal const u2y, amrex::ParticleReal const u2z,
            amrex::ParticleReal const m1, amrex::ParticleReal const m2,
            amrex::ParticleReal const w1, amrex::ParticleReal const w2,
            int const ion_level, amrex::Real const dt, amrex::Real const dV,
            int const pair_index, int const multiplier_ratio,
            index_type* AMREX_RESTRICT p_mask,
            amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
            amrex::ParticleReal* AMREX_RESTRICT p_product_data,
            amrex::RandomEngine const& engine) const
        {
            p_mask[pair_index] = index_type(0);

            // Ions that are fully ionized cannot be ionized further
            if (ion_level >= m_atomic_number) { return; }

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
            double const E_kin_ion_frame = E_COM*(E_COM + 2.0*(m1_d + m2_d)*c2)/(2.0*m2_d*c2);
            auto const E_kin_ion_frame_eV = static_cast<amrex::ParticleReal>(
                E_kin_ion_frame/PhysConst::q_e_v<double>);

            // Occupied subshells of the ion in its current charge state
            int const subshell_start = m_subshell_offsets[ion_level];
            int const n_subshells = m_subshell_offsets[ion_level+1] - subshell_start;

            amrex::ParticleReal const sigma = LotzIonizationCrossSection(
                E_kin_ion_frame_eV, n_subshells,
                m_occupation + subshell_start, m_binding_energy_eV + subshell_start);

            if (sigma <= amrex::ParticleReal(0.)) { return; }

            amrex::ParticleReal const w_min = amrex::min(w1, w2);
            amrex::ParticleReal const w_max = amrex::max(w1, w2);

            // Probability of ionization, following Higginson et al., JCP 388, 439-453 (2019)
            amrex::ParticleReal const exponent = multiplier_ratio * lab_to_COM_factor
                * w_max * sigma * v_rel_COM * dt / dV;

            // In principle the probability is 1 - exp(-exponent), but std::expm1 is used
            // since it maintains correctness for a small exponent.
            amrex::ParticleReal const probability = -std::expm1(-exponent);

            if (amrex::Random(engine) < probability)
            {
                p_mask[pair_index] = index_type(1);
                p_pair_reaction_weight[pair_index] = w_min;
                // The binding energies are sorted in increasing order: the first one
                // is the ionization energy of the ion
                p_product_data[pair_index] = m_binding_energy_eV[subshell_start];
            }
        }

        bool m_computeSpeciesDensities = false;
        bool m_computeSpeciesTemperatures = false;
        bool m_need_product_data = true;

        // Atomic number of the element of the ions
        int m_atomic_number = 0;
        // Index of the ionization level in the runtime integer attributes of the ions
        int m_ionization_level_comp = 0;
        // For each charge state, index of its first occupied subshell in the arrays below
        // (of size m_atomic_number + 1)
        int const* m_subshell_offsets = nullptr;
        // Number of electrons in each occupied subshell
        amrex::ParticleReal const* m_occupation = nullptr;
        // Binding energy of each occupied subshell [eV]
        amrex::ParticleReal const* m_binding_energy_eV = nullptr;
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

    amrex::Gpu::DeviceVector<int> m_subshell_offsets;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_occupation;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_binding_energy_eV;

    Executor m_exe;
};

#endif // WARPX_ELECTRON_IMPACT_IONIZATION_FUNC_H_
