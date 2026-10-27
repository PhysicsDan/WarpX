/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ElectronImpactIonizationFunc.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_ParmParse.H>

#include <string>

ElectronImpactIonizationFunc::ElectronImpactIonizationFunc (
    std::string const& collision_name, MultiParticleContainer const * const mypc,
    bool const isSameSpecies)
    : m_atomic_data(collision_name, mypc)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!isSameSpecies,
        "ElectronImpactIonizationFunc: the two colliding species must be different");

    // The second species is the target ion species
    auto const species_names =
        BinaryCollisionUtils::get_electron_impact_ionization_species(collision_name, mypc);
    m_ion_species = &mypc->GetParticleContainerFromName(species_names[1]);

    const amrex::ParmParse pp_collision_name(collision_name);
    pp_collision_name.query("split_macroparticles", m_exe.m_split_macroparticles);
    pp_collision_name.query_enum_sloppy("ejected_electron_energy_model",
                                        m_exe.m_ejected_electron_energy_model, "-_");

    // The kernels access the atomic data via the pointers of this light-weight structure
    m_exe.m_atomic_data = m_atomic_data.data();
}
