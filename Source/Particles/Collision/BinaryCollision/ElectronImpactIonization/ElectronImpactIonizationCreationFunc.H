/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRON_IMPACT_IONIZATION_CREATION_FUNC_H_
#define WARPX_ELECTRON_IMPACT_IONIZATION_CREATION_FUNC_H_

#include "ElectronImpactIonizationInitializeMomentum.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/ParticleCreation/DefaultInitialization.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX_DenseBins.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_Scan.H>
#include <AMReX_Vector.H>

#include <string>

/**
 * \brief This functor creates the particles that result from electron-impact ionization
 * events, and sets their properties (position, momentum, weight and ionization level).
 *
 * Each ionization event between an incident electron and an ion, of weight
 * p_pair_reaction_weight (which has already been subtracted from the weight of both
 * colliding macroparticles), creates three new macroparticles of that weight,
 * in the three product slots:
 *  - Slot 0: the incident electron species, for the scattered incident electron,
 *    created at the position of the incident electron.
 *  - Slot 1: the ion species, for the ion, whose ionization level is increased by one,
 *    created at the position of the ion.
 *  - Slot 2: the species of the ejected electrons, created at the position of the ion,
 *    which ensures local charge conservation.
 * This approach (also used for ionization in DSMC collisions) handles colliding
 * macroparticles with different weights, while conserving the energy and momentum.
 */
class ElectronImpactIonizationCreationFunc
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
     * \brief Default constructor of the ElectronImpactIonizationCreationFunc class.
     */
    ElectronImpactIonizationCreationFunc () = default;

    /**
     * \brief Constructor of the ElectronImpactIonizationCreationFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     */
    ElectronImpactIonizationCreationFunc (std::string const& collision_name,
                                          MultiParticleContainer const * mypc);

    /**
     * \brief operator() of the ElectronImpactIonizationCreationFunc class. It creates the new
     * particles from the electron-impact ionization events of a tile, and sets their properties.
     *
     * @param[in] n_total_pairs how many binary collisions have been performed in this tile
     * @param[in] ptile1 the particle tile of the incident electrons
     * @param[in] ptile2 the particle tile of the ions
     * @param[in] pc_products the product particle containers: the incident electron species,
     *            the ion species and the species of the ejected electrons
     * @param[in,out] tile_products array containing the tiles of the product particles
     * @param[in] m1 mass of the incident electrons
     * @param[in] m2 mass of the ions
     * @param[in] p_mask a mask that is 1 if the binary collision has resulted in an
     *            ionization event, 0 otherwise
     * @param[in,out] products_np array storing the number of existing particles in the
     *            tiles of the product species, before the new particles are added
     * @param[in] copy_species1 array of functors used to copy data from the incident electron
     *            species to the product species
     * @param[in] copy_species2 array of functors used to copy data from the ion species to
     *            the product species
     * @param[in] p_pair_indices_1 array that stores the indices of the incident electrons
     *            that were used in the binary collisions
     * @param[in] p_pair_indices_2 array that stores the indices of the ions that were used
     *            in the binary collisions
     * @param[in] p_pair_reaction_weight array that stores the weight of the product particles
     * @param[in] p_product_data array that stores the ionization energy of the ions [eV]
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

        // Compute the offset of each ionization event in the product tiles
        amrex::Gpu::DeviceVector<index_type> offsets(n_total_pairs);
        index_type* AMREX_RESTRICT p_offsets = offsets.dataPtr();
        index_type const n_events = amrex::Scan::ExclusiveSum(n_total_pairs, p_mask, p_offsets);

        // Each ionization event creates one particle in each product species.
        // Note that the colliding species are also product species, so that some of the
        // tiles below are those of the colliding species (and the same tile can appear twice
        // if the ejected electrons are added to the incident electron species). The tiles are
        // resized one after the other, and products_np keeps track of the positions at which
        // the new particles of each product slot are stored.
        amrex::Vector<int> num_added_vec(m_num_product_species, static_cast<int>(n_events));
        for (int i = 0; i < m_num_product_species; i++)
        {
            products_np[i] = tile_products[i]->numParticles();
            tile_products[i]->resize(products_np[i] + n_events);
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

        EjectedElectronEnergyModel const ejected_electron_energy_model =
            m_ejected_electron_energy_model;

        amrex::ParallelForRNG(n_total_pairs,
        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
        {
            if (p_mask[i] == 0) { return; }

            auto const i1 = static_cast<int>(p_pair_indices_1[i]);
            auto const i2 = static_cast<int>(p_pair_indices_2[i]);
            amrex::ParticleReal const weight = p_pair_reaction_weight[i];

            // Index of the new particle in each product slot
            auto const idx_electron = static_cast<int>(products_np_data[0] + p_offsets[i]);
            auto const idx_ion = static_cast<int>(products_np_data[1] + p_offsets[i]);
            auto const idx_ejected = static_cast<int>(products_np_data[2] + p_offsets[i]);

            // The scattered electron is created at the position of the incident electron,
            // while the ion and the ejected electron are created at the position of the ion
            copy_species1[0](soa_products_data[0], soa_1, i1, idx_electron, engine);
            copy_species2[1](soa_products_data[1], soa_2, i2, idx_ion, engine);
            copy_species2[2](soa_products_data[2], soa_2, i2, idx_ejected, engine);

            soa_products_data[0].m_rdata[PIdx::w][idx_electron] = weight;
            soa_products_data[1].m_rdata[PIdx::w][idx_ion] = weight;
            soa_products_data[2].m_rdata[PIdx::w][idx_ejected] = weight;

            // Ionization energy of the ion, converted from eV to J
            amrex::ParticleReal const ionization_energy = p_product_data[i]*PhysConst::q_e;

            ElectronImpactIonizationInitializeMomentum(
                soa_1.m_rdata[PIdx::ux][i1], soa_1.m_rdata[PIdx::uy][i1],
                soa_1.m_rdata[PIdx::uz][i1],
                soa_2.m_rdata[PIdx::ux][i2], soa_2.m_rdata[PIdx::uy][i2],
                soa_2.m_rdata[PIdx::uz][i2],
                m1, m2, ionization_energy, ejected_electron_energy_model,
                soa_products_data[0].m_rdata[PIdx::ux][idx_electron],
                soa_products_data[0].m_rdata[PIdx::uy][idx_electron],
                soa_products_data[0].m_rdata[PIdx::uz][idx_electron],
                soa_products_data[1].m_rdata[PIdx::ux][idx_ion],
                soa_products_data[1].m_rdata[PIdx::uy][idx_ion],
                soa_products_data[1].m_rdata[PIdx::uz][idx_ion],
                soa_products_data[2].m_rdata[PIdx::ux][idx_ejected],
                soa_products_data[2].m_rdata[PIdx::uy][idx_ejected],
                soa_products_data[2].m_rdata[PIdx::uz][idx_ejected],
                engine);
        });

        // Initialize the user runtime components
        for (int i = 0; i < m_num_product_species; i++)
        {
            int const start_index = static_cast<int>(products_np[i]);
            int const stop_index = static_cast<int>(products_np[i] + n_events);
            ParticleCreation::DefaultInitializeRuntimeAttributes(*tile_products[i],
                                                 *pc_products[i], start_index, stop_index);
        }

        // Set the ionization level of the new ions to that of the ions they come from,
        // plus one. (This is done after the default initialization of the runtime attributes,
        // which sets the ionization level of new particles to the initial ionization level.)
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

        amrex::Gpu::synchronize();

        return num_added_vec;
    }

private:
    // Number of product species: incident electrons, ions and ejected electrons
    int m_num_product_species = 3;

    // Model for the kinetic energy of the ejected electrons
    EjectedElectronEnergyModel m_ejected_electron_energy_model =
        EjectedElectronEnergyModel::Default;
};

#endif // WARPX_ELECTRON_IMPACT_IONIZATION_CREATION_FUNC_H_
