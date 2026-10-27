/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ElectronImpactIonizationCreationFunc.H"

#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_ParmParse.H>

#include <string>

ElectronImpactIonizationCreationFunc::ElectronImpactIonizationCreationFunc (
    std::string const& collision_name, MultiParticleContainer const * const mypc)
    : m_atomic_data(collision_name, mypc)
{
    const amrex::ParmParse pp_collision_name(collision_name);
    pp_collision_name.query("split_macroparticles", m_split_macroparticles);
    pp_collision_name.query_enum_sloppy("ejected_electron_energy_model",
                                        m_ejected_electron_energy_model, "-_");
}
