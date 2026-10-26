/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ElectronImpactIonizationCreationFunc.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_ParmParse.H>

#include <string>

ElectronImpactIonizationCreationFunc::ElectronImpactIonizationCreationFunc (
    std::string const& collision_name, MultiParticleContainer const * const mypc)
{
    // Check that the species involved are consistent with electron-impact ionization
    BinaryCollisionUtils::get_electron_impact_ionization_species(collision_name, mypc);

    const amrex::ParmParse pp_collision_name(collision_name);
    pp_collision_name.query_enum_sloppy("ejected_electron_energy_model",
                                        m_ejected_electron_energy_model, "-_");
}
