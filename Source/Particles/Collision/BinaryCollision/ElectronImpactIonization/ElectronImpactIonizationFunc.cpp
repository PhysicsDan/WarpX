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
#include "Utils/Physics/IonizationEnergiesTable.H"
#include "Utils/TextMsg.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <string>

namespace
{
    /**
     * \brief Number of electrons that can occupy each subshell, with the subshells listed
     * in the order in which they are filled in the ground state of atoms and ions,
     * according to the Madelung rule:
     * 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p
     */
    constexpr std::array<int, 19> madelung_subshell_capacities = {
        2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 6};
}

ElectronImpactIonizationFunc::ElectronImpactIonizationFunc (
    std::string const& collision_name, MultiParticleContainer const * const mypc,
    bool const isSameSpecies)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!isSameSpecies,
        "ElectronImpactIonizationFunc: the two colliding species must be different");

    // The second species is the target ion species
    auto const species_names =
        BinaryCollisionUtils::get_electron_impact_ionization_species(collision_name, mypc);
    m_ion_species = &mypc->GetParticleContainerFromName(species_names[1]);

    // Get the atomic number and the ionization energies of the element of the ions
    std::string physical_element;
    const amrex::ParmParse pp_ion_species_name(species_names[1]);
    pp_ion_species_name.get("physical_element", physical_element);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        utils::physics::ion_map_ids.count(physical_element) == 1,
        "ElectronImpactIonizationFunc: unknown physical element " + physical_element);
    int const ion_element_id = utils::physics::ion_map_ids.at(physical_element);
    int const atomic_number = utils::physics::ion_atomic_numbers[ion_element_id];
    int const energy_offset = utils::physics::ion_energy_offsets[ion_element_id];
    amrex::Vector<amrex::ParticleReal> ionization_energies(atomic_number);
    for (int i = 0; i < atomic_number; ++i) {
        ionization_energies[i] = static_cast<amrex::ParticleReal>(
            utils::physics::table_ionization_energies[i + energy_offset]);
    }

    // For each charge state, compute the occupation and the binding energy of each occupied
    // subshell, which are used to compute the ionization cross section.
    // The electrons fill the subshells in the order given by the Madelung rule.
    // The binding energy of a subshell is estimated as the ionization energy of the charge
    // state in which this subshell is the outermost occupied subshell, i.e. the charge state
    // obtained by removing all the electrons in the subshells filled after it.
    // The subshells are stored from the outermost to the innermost, i.e. by increasing
    // binding energy, so that the first binding energy is the ionization energy of the ion.
    amrex::Vector<int> h_subshell_offsets(atomic_number + 1);
    amrex::Vector<amrex::ParticleReal> h_occupation;
    amrex::Vector<amrex::ParticleReal> h_binding_energy_eV;
    for (int charge_state = 0; charge_state < atomic_number; ++charge_state) {
        h_subshell_offsets[charge_state] = static_cast<int>(h_occupation.size());

        int const n_bound_electrons = atomic_number - charge_state;
        amrex::Vector<int> occupation;
        amrex::Vector<int> n_electrons_up_to_subshell;
        int n_filled = 0;
        for (int const capacity : madelung_subshell_capacities) {
            if (n_filled == n_bound_electrons) { break; }
            int const n_in_subshell = std::min(capacity, n_bound_electrons - n_filled);
            n_filled += n_in_subshell;
            occupation.push_back(n_in_subshell);
            n_electrons_up_to_subshell.push_back(n_filled);
        }

        for (int s = static_cast<int>(occupation.size()) - 1; s >= 0; --s) {
            h_occupation.push_back(static_cast<amrex::ParticleReal>(occupation[s]));
            h_binding_energy_eV.push_back(
                ionization_energies[atomic_number - n_electrons_up_to_subshell[s]]);
        }
    }
    h_subshell_offsets[atomic_number] = static_cast<int>(h_occupation.size());

    // Transfer the data to the GPU device
    m_subshell_offsets.resize(h_subshell_offsets.size());
    m_occupation.resize(h_occupation.size());
    m_binding_energy_eV.resize(h_binding_energy_eV.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_subshell_offsets.begin(),
                          h_subshell_offsets.end(), m_subshell_offsets.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_occupation.begin(),
                          h_occupation.end(), m_occupation.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_binding_energy_eV.begin(),
                          h_binding_energy_eV.end(), m_binding_energy_eV.begin());
    amrex::Gpu::streamSynchronize();

    // The kernels access the data via these pointers
    m_exe.m_atomic_number = atomic_number;
    m_exe.m_subshell_offsets = m_subshell_offsets.dataPtr();
    m_exe.m_occupation = m_occupation.dataPtr();
    m_exe.m_binding_energy_eV = m_binding_energy_eV.dataPtr();
}
