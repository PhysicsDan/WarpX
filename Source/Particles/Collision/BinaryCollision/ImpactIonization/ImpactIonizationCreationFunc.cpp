/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ImpactIonizationCreationFunc.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_ParmParse.H>

#include <string>

ImpactIonizationCreationFunc::ImpactIonizationCreationFunc (
    std::string const& collision_name, MultiParticleContainer const * const mypc)
    : m_atomic_data(collision_name, mypc)
{
    auto const species = BinaryCollisionUtils::get_impact_ionization_species(collision_name, mypc);
    m_atomic_target = species.atomic_target;

    const amrex::ParmParse pp_collision_name(collision_name);
    pp_collision_name.query("split_macroparticles", m_split_macroparticles);
    pp_collision_name.query_enum_sloppy("ejected_electron_energy_model",
                                        m_ejected_electron_energy_model, "-_");
}
