/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ImpactIonizationFunc.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_ParmParse.H>

#include <string>

ImpactIonizationFunc::ImpactIonizationFunc (
    std::string const& collision_name, MultiParticleContainer const * const mypc,
    bool const isSameSpecies)
    : m_atomic_data(collision_name, mypc)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!isSameSpecies,
        "ImpactIonizationFunc: the two colliding species must be different");

    auto const species = BinaryCollisionUtils::get_impact_ionization_species(collision_name, mypc);
    m_target_species = &mypc->GetParticleContainerFromName(species.target);
    m_exe.m_atomic_target = species.atomic_target;

    const amrex::ParmParse pp_collision_name(collision_name);
    pp_collision_name.query("split_macroparticles", m_exe.m_split_macroparticles);
    pp_collision_name.query_enum_sloppy("ejected_electron_energy_model",
                                        m_exe.m_ejected_electron_energy_model, "-_");

    // The kernels access the atomic data via the pointers of this light-weight structure
    m_exe.m_atomic_data = m_atomic_data.data();
}
