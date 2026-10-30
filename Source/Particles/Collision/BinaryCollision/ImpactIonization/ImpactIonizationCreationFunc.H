/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_IMPACT_IONIZATION_CREATION_FUNC_H_
#define WARPX_IMPACT_IONIZATION_CREATION_FUNC_H_

#include "ImpactIonizationAtomicData.H"
#include "ImpactIonizationCrossSection.H"
#include "ImpactIonizationFunc.H"
#include "ImpactIonizationInitializeMomentum.H"

#include "Particles/ParticleCreation/DefaultInitialization.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_DenseBins.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_Scan.H>
#include <AMReX_Vector.H>

#include <string>

/**
 * \brief This functor creates the particles that result from impact ionization events, and
 * sets their properties (position, momentum, weight and ionization level).
 * The product slots are, in that order: the projectile species, the species of the ionized
 * targets (the target species itself for an atomic target, and the product ion species for a
 * molecular target) and the species of the ejected electrons.
 *
 * When the macroparticles are split, each ionization event between a projectile and a target,
 * of weight p_pair_reaction_weight (which has already been subtracted from the weight of both
 * colliding macroparticles), creates three new macroparticles of that weight:
 *  - Slot 0: the scattered projectile, created at the position of the projectile.
 *  - Slot 1: the ionized target, created at the position of the target (for an atomic target,
 *    its ionization level is increased by one).
 *  - Slot 2: the ejected electron, created at the position of the target, which ensures local
 *    charge conservation.
 * The momenta are computed with ImpactIonizationInitializeMomentum, which conserves the energy
 * (including the binding energy) and the momentum exactly.
 *
 * When the macroparticles are not split, the projectiles and the targets have already been
 * updated by ImpactIonizationFunc, and the ejected electrons are created (slot 2), at the
 * position of the target and with the weight of the target, one for each ionization of the
 * target. Their momenta are sampled with ImpactIonizationEjectedMomentum. For a molecular target,
 * which has been removed by ImpactIonizationFunc, a macroparticle of the product ion species is
 * also created (slot 1), with the position, momentum and weight of the target.
 */
class ImpactIonizationCreationFunc
{
    // Define shortcuts for frequently-used type names
    using ParticleType = typename WarpXParticleContainer::ParticleType;
    using ParticleTileType = typename WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = typename ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = typename ParticleBins::index_type;
    using SoaData_type = WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

public:
    /**
     * \brief Default constructor of the ImpactIonizationCreationFunc class.
     */
    ImpactIonizationCreationFunc () = default;

    /**
     * \brief Constructor of the ImpactIonizationCreationFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     */
    ImpactIonizationCreationFunc (std::string const& collision_name,
                                  MultiParticleContainer const * mypc);

    /**
     * \brief operator() of the ImpactIonizationCreationFunc class. It creates the new
     * particles from the impact ionization events of a tile, and sets their properties.
     *
     * @param[in] n_total_pairs how many binary collisions have been performed in this tile
     * @param[in] ptile1 the particle tile of the projectiles
     * @param[in] ptile2 the particle tile of the targets
     * @param[in] pc_products the product particle containers: the projectile species, the
     *            species of the ionized targets and the species of the ejected electrons
     * @param[in,out] tile_products array containing the tiles of the product particles
     * @param[in] m1 mass of the projectiles
     * @param[in] m2 mass of the targets
     * @param[in] p_mask when the macroparticles are split, 1 if the binary collision has
     *            resulted in an ionization event and 0 otherwise; when they are not split, the
     *            number of ionizations of the target of the pair and its initial ionization
     *            level (see ImpactIonizationFunc::mask_level_shift)
     * @param[in,out] products_np array storing the number of existing particles in the
     *            tiles of the product species, before the new particles are added
     * @param[in] copy_species1 array of functors used to copy data from the projectile
     *            species to the product species
     * @param[in] copy_species2 array of functors used to copy data from the target species to
     *            the product species
     * @param[in] p_pair_indices_1 array that stores the indices of the projectiles that were
     *            used in the binary collisions
     * @param[in] p_pair_indices_2 array that stores the indices of the targets that were used
     *            in the binary collisions
     * @param[in] p_pair_reaction_weight array that stores the weight of the product particles
     * @param[in] p_product_data when the macroparticles are not split, array that stores the
     *            kinetic energy of the projectile in the rest frame of the target [eV]
     * \return num_added number of particles added to each product species
     */
    AMREX_INLINE
    amrex::Vector<int> operator() (
        index_type const& n_total_pairs,
        ParticleTileType& ptile1, ParticleTileType& ptile2,
        amrex::Vector<WarpXParticleContainer*> const& pc_products,
        ParticleTileType** AMREX_RESTRICT tile_products,
        amrex::ParticleReal const m1, amrex::ParticleReal const m2,
        amrex::Vector<amrex::ParticleReal> const& /*products_mass*/,
        index_type const* AMREX_RESTRICT p_mask,
        amrex::Vector<index_type>& products_np,
        SmartCopy const* AMREX_RESTRICT copy_species1,
        SmartCopy const* AMREX_RESTRICT copy_species2,
        index_type const* AMREX_RESTRICT p_pair_indices_1,
        index_type const* AMREX_RESTRICT p_pair_indices_2,
        amrex::ParticleReal const* AMREX_RESTRICT p_pair_reaction_weight,
        amrex::ParticleReal const* AMREX_RESTRICT p_product_data) const
    {
        if (n_total_pairs == 0) { return amrex::Vector<int>(m_num_product_species, 0); }

        // Compute the offset of the new particles of each pair in the product tiles.
        // Each pair creates p_mask[i] & mask_count_bits particles in each product slot that
        // is filled (with the bit mask being irrelevant when the macroparticles are split).
        constexpr index_type mask_count_bits = ImpactIonizationFunc::mask_count_bits;
        constexpr int mask_level_shift = ImpactIonizationFunc::mask_level_shift;
        amrex::Gpu::DeviceVector<index_type> offsets(n_total_pairs);
        index_type* AMREX_RESTRICT p_offsets = offsets.dataPtr();
        index_type const n_new = amrex::Scan::PrefixSum<index_type>(n_total_pairs,
            [=] AMREX_GPU_DEVICE (index_type i) -> index_type {
                return p_mask[i] & mask_count_bits;
            },
            [=] AMREX_GPU_DEVICE (index_type i, index_type const& s) { p_offsets[i] = s; },
            amrex::Scan::Type::exclusive, amrex::Scan::retSum);

        // When the macroparticles are split, each ionization event creates one particle in
        // each product slot; otherwise, only the ejected electrons (slot 2) and, for a
        // molecular target, the product ions (slot 1) are created.
        // Note that the colliding species are also product species, so that some of the
        // tiles below are those of the colliding species (and the same tile can appear twice
        // if the ejected electrons are added to the projectile species). The tiles are
        // resized one after the other, and products_np keeps track of the positions at which
        // the new particles of each product slot are stored.
        bool const split_macroparticles = m_split_macroparticles;
        bool const atomic_target = m_atomic_target;
        amrex::Vector<int> num_added_vec(m_num_product_species, 0);
        for (int i = 0; i < m_num_product_species; i++)
        {
            if (split_macroparticles || i == 2 || (i == 1 && !atomic_target)) {
                num_added_vec[i] = static_cast<int>(n_new);
            }
            products_np[i] = tile_products[i]->numParticles();
            tile_products[i]->resize(products_np[i] + num_added_vec[i]);
        }

        // The particle data of the colliding species is accessed after the resize
        auto const soa_1 = ptile1.getParticleTileData();
        auto const soa_2 = ptile2.getParticleTileData();

        // Create necessary GPU vectors, that will be used in the kernel below
        amrex::Vector<SoaData_type> soa_products;
        for (int i = 0; i < m_num_product_species; i++)
        {
            soa_products.push_back(tile_products[i]->getParticleTileData());
        }
#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<SoaData_type> device_soa_products(m_num_product_species);
        amrex::Gpu::DeviceVector<index_type> device_products_np(m_num_product_species);

        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, soa_products.begin(),
                              soa_products.end(),
                              device_soa_products.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, products_np.begin(),
                              products_np.end(),
                              device_products_np.begin());

        amrex::Gpu::streamSynchronize();
        SoaData_type* AMREX_RESTRICT soa_products_data = device_soa_products.data();
        index_type const* AMREX_RESTRICT products_np_data = device_products_np.data();
#else
        SoaData_type* AMREX_RESTRICT soa_products_data = soa_products.data();
        index_type const* AMREX_RESTRICT products_np_data = products_np.data();
#endif

        ImpactIonization::AtomicData const atomic_data = m_atomic_data.data();
        EjectedElectronEnergyModel const ejected_electron_energy_model =
            m_ejected_electron_energy_model;
        // Ionization level of atomic targets (the charge state of molecular targets is 0)
        int const ion_level_comp = atomic_target ?
            pc_products[1]->GetIntCompIndex("ionizationLevel") - WarpXParticleContainer::NArrayInt
            : 0;

        amrex::ParallelForRNG(n_total_pairs,
        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
        {
            auto const n_created = static_cast<int>(p_mask[i] & mask_count_bits);
            if (n_created == 0) { return; }

            auto const i1 = static_cast<int>(p_pair_indices_1[i]);
            auto const i2 = static_cast<int>(p_pair_indices_2[i]);
            amrex::ParticleReal const weight = p_pair_reaction_weight[i];
            // Momentum per unit mass (gamma*v) of the projectile and of the target
            amrex::ParticleReal const u1x = soa_1.m_rdata[PIdx::ux][i1];
            amrex::ParticleReal const u1y = soa_1.m_rdata[PIdx::uy][i1];
            amrex::ParticleReal const u1z = soa_1.m_rdata[PIdx::uz][i1];
            amrex::ParticleReal const u2x = soa_2.m_rdata[PIdx::ux][i2];
            amrex::ParticleReal const u2y = soa_2.m_rdata[PIdx::uy][i2];
            amrex::ParticleReal const u2z = soa_2.m_rdata[PIdx::uz][i2];
            auto& soa_ejected = soa_products_data[2];

            if (split_macroparticles) {
                // Index of the new particle in each product slot
                auto const idx_projectile = static_cast<int>(products_np_data[0] + p_offsets[i]);
                auto const idx_ion = static_cast<int>(products_np_data[1] + p_offsets[i]);
                auto const idx_ejected = static_cast<int>(products_np_data[2] + p_offsets[i]);
                auto& soa_projectile = soa_products_data[0];
                auto& soa_ion = soa_products_data[1];

                // The scattered projectile is created at the position of the projectile, while
                // the ionized target and the ejected electron are created at the position of the
                // target
                copy_species1[0](soa_projectile, soa_1, i1, idx_projectile, engine);
                copy_species2[1](soa_ion, soa_2, i2, idx_ion, engine);
                copy_species2[2](soa_ejected, soa_2, i2, idx_ejected, engine);

                soa_projectile.m_rdata[PIdx::w][idx_projectile] = weight;
                soa_ion.m_rdata[PIdx::w][idx_ion] = weight;
                soa_ejected.m_rdata[PIdx::w][idx_ejected] = weight;

                int const charge_state = atomic_target ?
                    soa_2.m_runtime_idata[ion_level_comp][i2] : 0;
                double binding_energy;
                ImpactIonizationInitializeMomentum(
                    atomic_data, charge_state,
                    u1x, u1y, u1z, u2x, u2y, u2z, m1, m2, ejected_electron_energy_model,
                    soa_projectile.m_rdata[PIdx::ux][idx_projectile],
                    soa_projectile.m_rdata[PIdx::uy][idx_projectile],
                    soa_projectile.m_rdata[PIdx::uz][idx_projectile],
                    soa_ion.m_rdata[PIdx::ux][idx_ion],
                    soa_ion.m_rdata[PIdx::uy][idx_ion],
                    soa_ion.m_rdata[PIdx::uz][idx_ion],
                    soa_ejected.m_rdata[PIdx::ux][idx_ejected],
                    soa_ejected.m_rdata[PIdx::uy][idx_ejected],
                    soa_ejected.m_rdata[PIdx::uz][idx_ejected],
                    binding_energy, engine);
            } else {
                // For a molecular target, the product ion takes the place of the target, which
                // was removed (it is ionized at most once per collision step)
                if (!atomic_target) {
                    auto const idx_ion = static_cast<int>(products_np_data[1] + p_offsets[i]);
                    auto& soa_ion = soa_products_data[1];
                    copy_species2[1](soa_ion, soa_2, i2, idx_ion, engine);
                    soa_ion.m_rdata[PIdx::w][idx_ion] = weight;
                }
                // The target was ionized n_created times, starting from this ionization level
                auto const initial_level = static_cast<int>(p_mask[i] >> mask_level_shift);
                auto const E_kin = static_cast<double>(p_product_data[i]);
                for (int j = 0; j < n_created; ++j) {
                    auto const idx_ejected =
                        static_cast<int>(products_np_data[2] + p_offsets[i]) + j;
                    copy_species2[2](soa_ejected, soa_2, i2, idx_ejected, engine);
                    soa_ejected.m_rdata[PIdx::w][idx_ejected] = weight;
                    ImpactIonizationEjectedMomentum(
                        atomic_data, initial_level + j, E_kin,
                        u1x, u1y, u1z, u2x, u2y, u2z, ejected_electron_energy_model,
                        soa_ejected.m_rdata[PIdx::ux][idx_ejected],
                        soa_ejected.m_rdata[PIdx::uy][idx_ejected],
                        soa_ejected.m_rdata[PIdx::uz][idx_ejected],
                        engine);
                }
            }
        });

        // Initialize the user runtime components
        for (int i = 0; i < m_num_product_species; i++)
        {
            int const start_index = static_cast<int>(products_np[i]);
            int const stop_index = static_cast<int>(products_np[i]) + num_added_vec[i];
            ParticleCreation::DefaultInitializeRuntimeAttributes(*tile_products[i],
                                                 *pc_products[i], start_index, stop_index);
        }

        // When the macroparticles are split, set the ionization level of the new atomic targets
        // to that of the targets they come from, plus one. (This is done after the default
        // initialization of the runtime attributes, which sets the ionization level of new
        // particles to the initial ionization level.)
        if (split_macroparticles && atomic_target) {
            int const ion_level_index = pc_products[1]->GetIntCompIndex("ionizationLevel");
            int* AMREX_RESTRICT ion_level =
                tile_products[1]->GetStructOfArrays().GetIntData(ion_level_index).dataPtr();
            index_type const ion_np = products_np[1];
            amrex::ParallelFor(n_total_pairs,
            [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                if (p_mask[i] == 0) { return; }
                ion_level[ion_np + p_offsets[i]] = ion_level[p_pair_indices_2[i]] + 1;
            });
        }

        amrex::Gpu::synchronize();

        return num_added_vec;
    }

private:
    // Number of product species: projectiles, ionized targets and ejected electrons
    int m_num_product_species = 3;

    // Whether the target is atomic (with an ionization level) or molecular
    bool m_atomic_target = true;

    // Whether the colliding macroparticles are split in each ionization event
    bool m_split_macroparticles = false;

    // Model for the kinetic energy of the ejected electrons
    EjectedElectronEnergyModel m_ejected_electron_energy_model =
        EjectedElectronEnergyModel::Default;

    // Atomic data of the target
    ImpactIonizationAtomicData m_atomic_data;
};

#endif // WARPX_IMPACT_IONIZATION_CREATION_FUNC_H_
